"""Build DPO preference pairs from the policy's own rollouts.

For each task: sample K trajectories; chosen = a verified success (fewest
calls), rejected = a failure. With --expert-fallback, tasks the policy never
solves use the expert trajectory as chosen. Tasks where all K succeed or all
fail without a fallback give no pair (same idea as GRPO's zero-variance groups).

    python scripts/collect_pairs.py --model models/Qwen3-0.6B --lora runs/sft_qwen3_0.6b/lora.pt \
        --split train --n-tasks 600 --k 4 --out data/pairs_r1.jsonl --expert-fallback
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from cockpit.agent.format import ChatFormat  # noqa: E402
from cockpit.agent.batch_runner import run_episodes_batch  # noqa: E402
from cockpit.agent.runner import run_episode  # noqa: E402
from cockpit.baselines.planner import ExpertPolicy  # noqa: E402
from cockpit.tasks.splits import load_split  # noqa: E402
from cockpit.train.common import HFPolicy, assistant_prefix_for, load_model  # noqa: E402
from cockpit.train.lora import apply_lora, load_lora_state_dict  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--lora")
ap.add_argument("--split", default="train")
ap.add_argument("--n-tasks", type=int, default=600)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--max-new-tokens", type=int, default=256)
ap.add_argument("--expert-fallback", action="store_true")
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

model, tok = load_model(a.model, device=a.device)
if a.lora:
    apply_lora(model, dropout=0.0)
    load_lora_state_dict(model, torch.load(a.lora, map_location="cpu"))
model.eval()
fmt = ChatFormat(tok, assistant_prefix_for(a.model))
pol = HFPolicy(model, fmt, temperature=a.temperature, max_new_tokens=a.max_new_tokens)

tasks = load_split(f"data/{a.split}.jsonl")
random.Random(a.seed).shuffle(tasks)
stats = {"pairs": 0, "all_success": 0, "all_fail": 0, "expert_chosen": 0}


def clean(msgs):
    return [{k: v for k, v in m.items() if v is not None} for m in msgs]


with open(a.out, "w", encoding="utf-8") as f:
    for t in tasks[: a.n_tasks]:
        rs = run_episodes_batch([t] * a.k, pol)
        good = sorted([r for r in rs if r["verify"]["success"]], key=lambda r: r["verify"]["n_calls"])
        bad = [r for r in rs if not r["verify"]["success"]]
        if not bad:
            stats["all_success"] += 1
            continue
        if good:
            chosen = good[0]["messages"]
        elif a.expert_fallback:
            chosen = run_episode(t, ExpertPolicy())["messages"]
            stats["expert_chosen"] += 1
        else:
            stats["all_fail"] += 1
            continue
        f.write(json.dumps({"task_id": t["id"], "tier": t["tier"], "chosen": clean(chosen),
                            "rejected": clean(bad[0]["messages"]),
                            "rejected_reasons": bad[0]["verify"]["reasons"]}, ensure_ascii=False) + "\n")
        stats["pairs"] += 1
print(stats)
