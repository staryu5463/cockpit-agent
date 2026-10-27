"""Multi-turn GRPO with optional DAPO components, written from scratch.

One step:
  1. Sample B tasks; run G rollouts each in the real environment.
  2. Score each rollout with the verifier -> reward r_i.
  3. Group-relative advantage:  A_i = (r_i - mean_g) / (std_g + eps)   (std optional)
     DAPO dynamic sampling: drop groups whose rewards are all equal (A = 0 for all,
     so they carry no gradient) and keep sampling until B useful groups are found.
  4. Policy loss on the tokens the model actually generated (assistant turns only;
     tool responses are masked out):
         ratio = exp(logp - logp_old)
         L = -min(ratio * A, clip(ratio, 1 - eps_low, 1 + eps_high) * A) + beta * KL_k3
     DAPO clip-higher: eps_high > eps_low.
     Aggregation: "token" (DAPO, every token weighs the same) or "seq" (original GRPO,
     mean inside each sequence first).
  5. KL is against the base model = the same network with LoRA switched off.

    python -m cockpit.train.grpo --config configs/grpo_qwen3_1.7b.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from cockpit.agent.format import ChatFormat
from cockpit.agent.batch_runner import run_episodes_batch
from cockpit.agent.runner import run_episode
from cockpit.baselines.planner import ExpertPolicy
from cockpit.tasks.splits import load_split
from cockpit.train.common import (HFPolicy, assistant_prefix_for, load_model, pad_batch,
                                  token_logprobs)
from cockpit.train.lora import (apply_lora, load_lora_state_dict, lora_disabled, lora_parameters,
                                lora_state_dict)
from cockpit.train.reward import compute_reward

DEFAULTS = {
    "group_size": 8, "prompts_per_step": 4, "steps": 200, "lr": 1e-5,
    "clip_low": 0.2, "clip_high": 0.28,         # DAPO clip-higher; set equal for plain GRPO
    "beta_kl": 0.0,                             # DAPO drops KL; GRPO used ~0.04
    "loss_agg": "token",                        # "token" (DAPO) | "seq" (GRPO)
    "std_norm": True,                           # False = Dr.GRPO-style advantage
    "dynamic_sampling": True, "max_resample": 4,
    "truncation": "filter",                     # unfinished episodes: "filter" | "keep"
    "ppo_epochs": 1, "micro_batch": 2, "max_len": 3072,
    "temperature": 1.0, "top_p": 1.0, "max_new_tokens": 256,
    "reward_weights": {}, "tiers": None, "save_every": 25, "log_every": 1, "seed": 0,
}


class GRPOTrainer:
    def __init__(self, cfg):
        self.cfg = {**DEFAULTS, **cfg}
        c = self.cfg
        random.seed(c["seed"]); torch.manual_seed(c["seed"])
        self.out = Path(c["output_dir"]); self.out.mkdir(parents=True, exist_ok=True)
        self.model, self.tok = load_model(c["model"], c.get("dtype", "bfloat16"), c.get("device", "cuda"))
        self.fmt = ChatFormat(self.tok, c.get("assistant_prefix", assistant_prefix_for(c["model"])))
        apply_lora(self.model, r=c.get("lora_r", 16), alpha=c.get("lora_alpha", 32), dropout=0.0)
        if c.get("init_lora"):
            load_lora_state_dict(self.model, torch.load(c["init_lora"], map_location="cpu"))
        if c.get("gradient_checkpointing"):
            self.model.gradient_checkpointing_enable(); self.model.enable_input_require_grads()
        self.opt = torch.optim.AdamW(lora_parameters(self.model), lr=c["lr"], weight_decay=0.0)
        tasks = load_split(c["train_file"])
        if c["tiers"]:
            tasks = [t for t in tasks if t["tier"] in c["tiers"]]
        self.tasks = tasks
        self._expert_calls, self._expert_turns = {}, {}
        self.log = open(self.out / "log.jsonl", "a", encoding="utf-8")
        self.policy = HFPolicy(self.model, self.fmt, temperature=c["temperature"], top_p=c["top_p"],
                               max_new_tokens=c["max_new_tokens"])

    # ---------------------------------------------------------------- rollouts
    def expert_calls(self, task):
        if task["id"] not in self._expert_calls:
            self._expert_calls[task["id"]] = run_episode(task, ExpertPolicy())["verify"]["n_calls"]
        return self._expert_calls[task["id"]]

    def expert_turns(self, task):
        """Fewest turns the fair expert needs (parallel schedule when COCKPIT_PARALLEL=1)."""
        if task["id"] not in self._expert_turns:
            self._expert_turns[task["id"]] = run_episode(task, ExpertPolicy())["verify"]["turns"]
        return self._expert_turns[task["id"]]

    def rollout_group(self, task):
        """G rollouts of one task, generated in lock-step batches (batch_runner)."""
        self.model.eval()
        group = run_episodes_batch([task] * self.cfg["group_size"], self.policy)
        for res in group:
            res["reward"] = compute_reward(res["verify"], self.expert_calls(task), self.cfg["reward_weights"],
                                           self.expert_turns(task))
        return group

    def collect(self):
        c = self.cfg
        groups, stats = [], {"groups_sampled": 0, "groups_zero_std": 0}
        tries = 0
        while len(groups) < c["prompts_per_step"] and tries < c["prompts_per_step"] * c["max_resample"]:
            tries += 1
            task = random.choice(self.tasks)
            g = self.rollout_group(task)
            stats["groups_sampled"] += 1
            rs = [r["reward"] for r in g]
            if max(rs) - min(rs) < 1e-6:
                stats["groups_zero_std"] += 1
                if c["dynamic_sampling"]:
                    stats.setdefault("dropped", []).append(g)
                    continue
            groups.append(g)
        return groups, stats

    # ---------------------------------------------------------------- batches
    def build_samples(self, groups):
        c = self.cfg
        samples = []
        for g in groups:
            rs = [r["reward"] for r in g]
            mu = statistics.fmean(rs)
            sd = statistics.pstdev(rs) if len(rs) > 1 else 0.0
            for r in g:
                if c["truncation"] == "filter" and not r["verify"]["finished"]:
                    continue                                    # DAPO overlong filtering
                adv = (r["reward"] - mu) / (sd + 1e-6) if c["std_norm"] else r["reward"] - mu
                ids, mask, _ = self.fmt.encode(r["messages"])
                if len(ids) > c["max_len"]:
                    continue
                samples.append({"ids": ids, "mask": mask, "adv": adv})
        return samples

    @torch.no_grad()
    def precompute(self, samples):
        """old log-probs (for the ratio) and reference log-probs (for KL)."""
        self.model.eval()
        mb = self.cfg["micro_batch"]
        for i in range(0, len(samples), mb):
            chunk = samples[i:i + mb]
            ids, att, lm = pad_batch([s["ids"] for s in chunk], [s["mask"] for s in chunk], self.tok.pad_token_id)
            ids, att = ids.to(self.model.device), att.to(self.model.device)
            old, _ = token_logprobs(self.model, ids, att)
            ref = None
            if self.cfg["beta_kl"] > 0:
                with lora_disabled(self.model):
                    ref, _ = token_logprobs(self.model, ids, att)
            for j, s in enumerate(chunk):
                n = len(s["ids"]) - 1
                s["old"] = old[j, :n].cpu()
                s["ref"] = ref[j, :n].cpu() if ref is not None else None

    # ---------------------------------------------------------------- loss
    def loss_and_stats(self, chunk, n_tokens_total, n_seqs_total):
        c = self.cfg
        dev = self.model.device
        ids, att, lm = pad_batch([s["ids"] for s in chunk], [s["mask"] for s in chunk], self.tok.pad_token_id)
        ids, att, lm = ids.to(dev), att.to(dev), lm.to(dev)
        logp, ent = token_logprobs(self.model, ids, att)
        T = logp.shape[1]
        m = lm[:, 1:]                                           # mask aligned with targets
        old = torch.zeros_like(logp); ref = torch.zeros_like(logp)
        adv = torch.tensor([s["adv"] for s in chunk], device=dev, dtype=logp.dtype).unsqueeze(1)
        for j, s in enumerate(chunk):
            n = min(T, len(s["old"]))
            old[j, :n] = s["old"][:n].to(dev)
            if s["ref"] is not None:
                ref[j, :n] = s["ref"][:n].to(dev)
        ratio = torch.exp(logp - old)
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1 - c["clip_low"], 1 + c["clip_high"]) * adv
        per_tok = -torch.min(unclipped, clipped)
        kl = torch.zeros_like(per_tok)
        if c["beta_kl"] > 0:
            d = ref - logp
            kl = torch.exp(d) - d - 1                               # k3 estimator, >= 0
            per_tok = per_tok + c["beta_kl"] * kl
        if c["loss_agg"] == "token":
            loss = (per_tok * m).sum() / n_tokens_total
        else:
            loss = ((per_tok * m).sum(1) / m.sum(1).clamp(min=1)).sum() / n_seqs_total
        with torch.no_grad():
            clip_frac = (((ratio < 1 - c["clip_low"]) | (ratio > 1 + c["clip_high"])).float() * m).sum()
            st = {"tok": m.sum().item(), "ent": (ent * m).sum().item(), "kl": (kl * m).sum().item(),
                  "clip": clip_frac.item()}
        return loss, st

    # ---------------------------------------------------------------- step
    def step(self, it):
        c = self.cfg
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        groups, gstats = self.collect()
        flat = [r for g in groups for r in g] + [r for g in gstats.get("dropped", []) for r in g]
        samples = self.build_samples(groups)
        rec = {"step": it, "groups_used": len(groups), "groups_sampled": gstats["groups_sampled"],
               "frac_zero_std": round(gstats["groups_zero_std"] / max(1, gstats["groups_sampled"]), 3),
               "reward": round(statistics.fmean(r["reward"] for r in flat), 4) if flat else None,
               "success": round(statistics.fmean(r["verify"]["success"] for r in flat), 4) if flat else None,
               "format_err": round(statistics.fmean(r["verify"]["format_errors"] > 0 for r in flat), 4) if flat else None,
               "side_effect": round(statistics.fmean(bool(r["verify"]["side_effects"]) for r in flat), 4) if flat else None,
               "turns": round(statistics.fmean(r["verify"]["turns"] for r in flat), 2) if flat else None,
               "rollout_sec": round(time.time() - t0, 1)}
        if not samples:
            rec["note"] = "no trainable samples this step"
            self._write(rec)
            return rec
        # Generation leaves large KV-cache blocks in PyTorch's allocator. Give them back before the
        # update: on an 8 GB card the overflow otherwise spills into shared system memory (Windows
        # WDDM) and each step becomes several times slower instead of failing with OOM.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.precompute(samples)
        # Stay in eval mode: dropout would make logp != logp_old even before any update,
        # so the PPO ratio would not start at exactly 1 (clip_frac > 0 at ppo_epochs=1).
        self.model.eval()
        n_tok = sum(sum(s["mask"]) for s in samples)
        agg = {"tok": 0, "ent": 0, "kl": 0, "clip": 0}
        for _ in range(c["ppo_epochs"]):
            random.shuffle(samples)
            self.opt.zero_grad(set_to_none=True)
            for i in range(0, len(samples), c["micro_batch"]):
                loss, st = self.loss_and_stats(samples[i:i + c["micro_batch"]], n_tok, len(samples))
                loss.backward()
                for k in agg:
                    agg[k] += st[k]
            gn = torch.nn.utils.clip_grad_norm_(lora_parameters(self.model), 1.0)
            self.opt.step()
        rec.update(entropy=round(agg["ent"] / max(1, agg["tok"]), 4), kl=round(agg["kl"] / max(1, agg["tok"]), 5),
                   clip_frac=round(agg["clip"] / max(1, agg["tok"]), 4), grad_norm=round(float(gn), 4),
                   resp_tokens=round(agg["tok"] / c["ppo_epochs"] / len(samples), 1),
                   step_sec=round(time.time() - t0, 1))
        self._write(rec)
        return rec

    def _write(self, rec):
        print(json.dumps(rec, ensure_ascii=False))
        self.log.write(json.dumps(rec, ensure_ascii=False) + "\n"); self.log.flush()

    def latest_checkpoint(self):
        ck = sorted(self.out.glob("lora_step*.pt"), key=lambda p: int(p.stem.split("step")[1]))
        return (ck[-1], int(ck[-1].stem.split("step")[1])) if ck else (None, 0)

    def train(self):
        start = 1
        if self.cfg.get("resume"):
            path, done = self.latest_checkpoint()
            if path is not None:
                # optimizer moments are not saved: AdamW restarts from zero (noted in the log)
                load_lora_state_dict(self.model, torch.load(path, map_location="cpu"))
                start = done + 1
                self._write({"resume_from": str(path), "next_step": start})
        for it in range(start, self.cfg["steps"] + 1):
            self.step(it)
            if it % self.cfg["save_every"] == 0 or it == self.cfg["steps"]:
                torch.save(lora_state_dict(self.model), self.out / f"lora_step{it}.pt")
        json.dump(self.cfg, open(self.out / "config.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", default="{}")
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8"))
    ov = a.override
    cfg.update(json.load(open(ov, encoding="utf-8")) if ov.endswith(".json") else json.loads(ov))
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    GRPOTrainer(cfg).train()
