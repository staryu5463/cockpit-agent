"""Hard-task mining: find the tasks where the current policy is *uncertain*.

Sample K rollouts per task. Tasks solved K/K or 0/K give GRPO no signal (every
advantage is 0), so GRPO trains only on tasks with 0 < success rate < 1. The
same rollouts also yield DPO preference pairs (chosen = success, rejected =
failure; the expert stands in when the policy never succeeds).

Pool = regular training tasks (L3-L5) + "train_hard" tasks (dense near-tie POIs).

    python scripts/mine_hard.py --model models/Qwen3-0.6B --lora runs/sft_qwen3_0.6b/lora.pt \
        --n-regular 200 --n-hard 200 --k 8 --out-tasks data/rl_hard.jsonl --out-pairs data/pairs_r1.jsonl
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from cockpit.agent.batch_runner import run_episodes_batch  # noqa: E402
from cockpit.agent.format import ChatFormat  # noqa: E402
from cockpit.agent.runner import run_episode  # noqa: E402
from cockpit.baselines.planner import ExpertPolicy  # noqa: E402
from cockpit.tasks.splits import build_split, load_split  # noqa: E402
from cockpit.train.common import HFPolicy, assistant_prefix_for, load_model  # noqa: E402
from cockpit.train.lora import apply_lora, load_lora_state_dict  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--lora")
ap.add_argument("--n-regular", type=int, default=200)
ap.add_argument("--n-hard", type=int, default=200)
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--max-new-tokens", type=int, default=128)
ap.add_argument("--out-tasks", required=True)
ap.add_argument("--out-pairs", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

model, tok = load_model(a.model)
if a.lora:
    apply_lora(model, dropout=0.0)
    load_lora_state_dict(model, torch.load(a.lora, map_location="cpu"))
model.eval()
pol = HFPolicy(model, ChatFormat(tok, assistant_prefix_for(a.model)), temperature=a.temperature,
               max_new_tokens=a.max_new_tokens)

rng = random.Random(a.seed)
regular = [t for t in load_split("data/train.jsonl") if t["tier"] in ("L3", "L4", "L5")]
rng.shuffle(regular)
pool = [dict(t, pool="regular") for t in regular[: a.n_regular]]
pool += [dict(t, pool="hard") for t in build_split("train_hard", a.n_hard, seed=a.seed + 1)]


def clean(msgs):
    return [{k: v for k, v in m.items() if v is not None} for m in msgs]


rates = defaultdict(list)
reasons = Counter()
n_mixed = n_pairs = 0
with open(a.out_tasks, "w", encoding="utf-8") as ft, open(a.out_pairs, "w", encoding="utf-8") as fp:
    for i, t in enumerate(pool):
        rs = run_episodes_batch([t] * a.k, pol)
        ok = [r for r in rs if r["verify"]["success"]]
        bad = [r for r in rs if not r["verify"]["success"]]
        p = len(ok) / a.k
        rates[(t["pool"], t["tier"])].append(p)
        for r in bad:
            reasons.update(x.split(":")[0] for x in r["verify"]["reasons"])
        if 0 < p < 1:
            ft.write(json.dumps(t, ensure_ascii=False) + "\n")
            n_mixed += 1
        if bad:
            chosen = (sorted(ok, key=lambda r: r["verify"]["n_calls"])[0]["messages"] if ok
                      else run_episode(t, ExpertPolicy())["messages"])
            fp.write(json.dumps({"task_id": t["id"], "tier": t["tier"], "pool": t["pool"],
                                 "chosen": clean(chosen), "rejected": clean(bad[0]["messages"]),
                                 "rejected_reasons": bad[0]["verify"]["reasons"],
                                 "chosen_from": "policy" if ok else "expert"}, ensure_ascii=False) + "\n")
            n_pairs += 1
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(pool)} mixed={n_mixed} pairs={n_pairs}", flush=True)

stats = {"tasks": len(pool), "mixed (GRPO)": n_mixed, "pairs (DPO)": n_pairs,
         "success_rate_by_pool_tier": {f"{k[0]}/{k[1]}": round(sum(v) / len(v), 3) for k, v in sorted(rates.items())},
         "share_all_success": round(sum(p == 1 for v in rates.values() for p in v) / len(pool), 3),
         "share_all_fail": round(sum(p == 0 for v in rates.values() for p in v) / len(pool), 3),
         "failure_reasons": dict(reasons.most_common())}
print(json.dumps(stats, ensure_ascii=False, indent=2))
json.dump(stats, open(Path(a.out_tasks).with_suffix(".stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
