"""LoRA SFT on expert trajectories.

Loss is computed only on assistant tokens (the tool calls the model must
produce). System prompt, user text and tool responses are context, not targets.

    python -m cockpit.train.sft --config configs/sft_qwen3_0.6b.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from cockpit.agent.format import ChatFormat
from cockpit.agent.runner import run_episode
from cockpit.baselines.planner import ExpertPolicy
from cockpit.tasks.splits import load_split
from cockpit.train.common import assistant_prefix_for, load_model, pad_batch
from cockpit.train.lora import apply_lora, count_trainable, lora_parameters, lora_state_dict


def build_examples(tasks, fmt, max_len):
    ex, skipped = [], 0
    for t in tasks:
        res = run_episode(t, ExpertPolicy())
        if not res["verify"]["success"]:
            skipped += 1
            continue
        ids, mask, _ = fmt.encode(res["messages"])
        if len(ids) > max_len:
            skipped += 1
            continue
        ex.append((ids, mask))
    return ex, skipped


def masked_ce(model, ids, att, lm):
    """Mean cross-entropy over assistant tokens (chunked log-softmax, see common.py)."""
    from cockpit.train.common import token_logprobs
    logp, _ = token_logprobs(model, ids, att)
    m = lm[:, 1:]
    return -(logp * m).sum() / m.sum().clamp(min=1)


def main(cfg):
    torch.manual_seed(cfg.get("seed", 0)); random.seed(cfg.get("seed", 0))
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(cfg["model"], cfg.get("dtype", "bfloat16"), cfg.get("device", "cuda"))
    fmt = ChatFormat(tok, cfg.get("assistant_prefix", assistant_prefix_for(cfg["model"])))
    apply_lora(model, r=cfg.get("lora_r", 16), alpha=cfg.get("lora_alpha", 32))
    if cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    print(f"trainable params: {count_trainable(model):,}")

    train_tasks = load_split(cfg["train_file"])[: cfg.get("max_train", None)]
    dev_tasks = load_split(cfg["dev_file"])[: cfg.get("max_dev", 200)]
    train, sk = build_examples(train_tasks, fmt, cfg.get("max_len", 2048))
    dev, _ = build_examples(dev_tasks, fmt, cfg.get("max_len", 2048))
    print(f"train examples {len(train)} (skipped {sk}), dev {len(dev)}")

    opt = torch.optim.AdamW(lora_parameters(model), lr=cfg.get("lr", 2e-4), weight_decay=0.0)
    bs, accum, epochs = cfg.get("batch_size", 2), cfg.get("grad_accum", 8), cfg.get("epochs", 2)
    total = math.ceil(len(train) / bs / accum) * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, cfg.get("warmup_steps", 20))) *
        0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    log = open(out / "log.jsonl", "w", encoding="utf-8")
    dev_ = model.device
    step, t0 = 0, time.time()
    for ep in range(epochs):
        random.shuffle(train)
        model.train()
        for i in range(0, len(train), bs):
            batch = train[i:i + bs]
            ids, att, lm = pad_batch([b[0] for b in batch], [b[1] for b in batch], tok.pad_token_id)
            loss = masked_ce(model, ids.to(dev_), att.to(dev_), lm.to(dev_)) / accum
            loss.backward()
            if (i // bs + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(lora_parameters(model), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % cfg.get("log_every", 10) == 0:
                    rec = {"step": step, "epoch": ep, "loss": loss.item() * accum,
                           "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)}
                    print(rec); log.write(json.dumps(rec) + "\n"); log.flush()
        dl = evaluate_loss(model, dev, tok, bs)
        rec = {"epoch": ep, "dev_loss": dl}
        print(rec); log.write(json.dumps(rec) + "\n"); log.flush()
        torch.save(lora_state_dict(model), out / "lora.pt")
    json.dump(cfg, open(out / "config.json", "w"), indent=2, ensure_ascii=False)
    print("saved", out / "lora.pt")


@torch.no_grad()
def evaluate_loss(model, data, tok, bs):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(data), bs):
        batch = data[i:i + bs]
        ids, att, lm = pad_batch([b[0] for b in batch], [b[1] for b in batch], tok.pad_token_id)
        tot += masked_ce(model, ids.to(model.device), att.to(model.device), lm.to(model.device)).item()
        n += 1
    model.train()
    return tot / max(1, n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", default="{}", help='JSON, e.g. {"epochs": 1}')
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8"))
    ov = a.override
    cfg.update(json.load(open(ov, encoding="utf-8")) if ov.endswith(".json") else json.loads(ov))
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    main(cfg)
