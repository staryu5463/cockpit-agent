"""DPO on preference pairs built from the agent's own rollouts.

Preference data (scripts/collect_pairs.py): for one task, sample K trajectories
with the current policy; a verified success is "chosen", a failure "rejected".
If the policy never succeeds, the expert trajectory can stand in as chosen.

Loss (Rafailov et al. 2023, arXiv 2305.18290), on assistant tokens only:

    margin = beta * [ (log pi(c) - log ref(c)) - (log pi(r) - log ref(r)) ]
    L      = -log sigmoid(margin)

log pi(.) sums log-probs over the tokens the policy generated (tool responses
are masked, as in SFT / GRPO). The reference is the starting policy (usually
the SFT checkpoint); its log-probs are computed once before training.

Why DPO next to GRPO: same data source (rollouts + verifier), no online
sampling during the update, cheaper per step. Comparing the two on the same
budget answers "is online RL worth it here?".

    python -m cockpit.train.dpo --config configs/dpo_qwen3_0.6b.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from cockpit.agent.format import ChatFormat
from cockpit.train.common import assistant_prefix_for, load_model, pad_batch, token_logprobs
from cockpit.train.lora import apply_lora, load_lora_state_dict, lora_parameters, lora_state_dict

DEFAULTS = {"beta": 0.1, "lr": 5e-6, "epochs": 1, "micro_batch": 1, "grad_accum": 8,
            "max_len": 3072, "log_every": 5, "seed": 0}


def seq_logp(model, fmt, seqs):
    """Sum of log-probs over assistant tokens for each (ids, mask)."""
    ids, att, lm = pad_batch([s[0] for s in seqs], [s[1] for s in seqs], fmt.tok.pad_token_id)
    dev = model.device
    lp, _ = token_logprobs(model, ids.to(dev), att.to(dev))
    return (lp * lm[:, 1:].to(dev)).sum(1)


def load_pairs(path, fmt, max_len):
    pairs = []
    for line in open(path, encoding="utf-8"):
        p = json.loads(line)
        c = fmt.encode(p["chosen"])[:2]
        r = fmt.encode(p["rejected"])[:2]
        if max(len(c[0]), len(r[0])) <= max_len:
            pairs.append({"task_id": p["task_id"], "c": c, "r": r})
    return pairs


class DPOTrainer:
    def __init__(self, cfg):
        self.cfg = c = {**DEFAULTS, **cfg}
        random.seed(c["seed"]); torch.manual_seed(c["seed"])
        self.out = Path(c["output_dir"]); self.out.mkdir(parents=True, exist_ok=True)
        self.model, tok = load_model(c["model"], c.get("dtype", "bfloat16"), c.get("device", "cuda"))
        self.fmt = ChatFormat(tok, c.get("assistant_prefix", assistant_prefix_for(c["model"])))
        apply_lora(self.model, r=c.get("lora_r", 16), alpha=c.get("lora_alpha", 32), dropout=0.0)
        if c.get("init_lora"):
            load_lora_state_dict(self.model, torch.load(c["init_lora"], map_location="cpu"))
        if c.get("gradient_checkpointing"):
            self.model.gradient_checkpointing_enable(); self.model.enable_input_require_grads()
        self.model.eval()                              # no dropout: logp must be deterministic
        self.opt = torch.optim.AdamW(lora_parameters(self.model), lr=c["lr"], weight_decay=0.0)

    @torch.no_grad()
    def precompute_ref(self, pairs):
        for p in pairs:
            p["ref_c"] = seq_logp(self.model, self.fmt, [p["c"]]).item()
            p["ref_r"] = seq_logp(self.model, self.fmt, [p["r"]]).item()

    def loss(self, batch):
        b = self.cfg["beta"]
        pc = seq_logp(self.model, self.fmt, [p["c"] for p in batch])
        pr = seq_logp(self.model, self.fmt, [p["r"] for p in batch])
        dev = pc.device
        rc = torch.tensor([p["ref_c"] for p in batch], device=dev)
        rr = torch.tensor([p["ref_r"] for p in batch], device=dev)
        margin = b * ((pc - rc) - (pr - rr))
        return -F.logsigmoid(margin).mean(), margin.detach()

    def train(self, pairs):
        c = self.cfg
        t0 = time.time()
        self.precompute_ref(pairs)
        log = open(self.out / "log.jsonl", "a", encoding="utf-8")
        step = 0
        for ep in range(c["epochs"]):
            random.shuffle(pairs)
            margins = []
            for i in range(0, len(pairs), c["micro_batch"]):
                loss, m = self.loss(pairs[i:i + c["micro_batch"]])
                (loss / c["grad_accum"]).backward()
                margins += m.tolist()
                if (i // c["micro_batch"] + 1) % c["grad_accum"] == 0:
                    torch.nn.utils.clip_grad_norm_(lora_parameters(self.model), 1.0)
                    self.opt.step(); self.opt.zero_grad(set_to_none=True); step += 1
                    if step % c["log_every"] == 0:
                        rec = {"step": step, "epoch": ep, "loss": round(loss.item(), 4),
                               "reward_acc": round(sum(x > 0 for x in margins) / len(margins), 3),
                               "margin": round(sum(margins) / len(margins), 4), "sec": round(time.time() - t0, 1)}
                        print(rec); log.write(json.dumps(rec) + "\n"); log.flush()
                        margins = []
        torch.save(lora_state_dict(self.model), self.out / "lora.pt")
        json.dump(c, open(self.out / "config.json", "w"), indent=2, ensure_ascii=False)
        print("saved", self.out / "lora.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", default="{}")
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8"))
    ov = a.override
    cfg.update(json.load(open(ov, encoding="utf-8")) if ov.endswith(".json") else json.loads(ov))
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    tr = DPOTrainer(cfg)
    pairs = load_pairs(cfg["pairs_file"], tr.fmt, tr.cfg["max_len"])
    print(f"{len(pairs)} preference pairs")
    tr.train(pairs)


if __name__ == "__main__":
    main()
