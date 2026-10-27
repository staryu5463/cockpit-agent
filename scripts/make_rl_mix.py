"""GRPO task file for the efficiency experiments: the mined hard tasks plus training tasks
whose fewest-turn (parallel) solution is shorter than the one-call-per-turn solution, so an
efficiency reward has something to teach.

    COCKPIT_PARALLEL=1 python scripts/make_rl_mix.py --n-parallel 100 --out data/rl_mix.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cockpit.agent.runner import run_episode  # noqa: E402
from cockpit.baselines.planner import ExpertPolicy  # noqa: E402
from cockpit.tasks.splits import load_split  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--hard", default="data/rl_hard.jsonl")
ap.add_argument("--train", default="data/train.jsonl")
ap.add_argument("--n-parallel", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="data/rl_mix.jsonl")
a = ap.parse_args()

hard = load_split(a.hard)
seen = {t["id"] for t in hard}
pool = [t for t in load_split(a.train) if t["id"] not in seen]
random.Random(a.seed).shuffle(pool)
extra = []
for t in pool:
    seq = run_episode(t, ExpertPolicy(parallel=False))["verify"]["turns"]
    par = run_episode(t, ExpertPolicy(parallel=True))["verify"]["turns"]
    if par < seq:
        extra.append(t)
    if len(extra) == a.n_parallel:
        break
with open(a.out, "w", encoding="utf-8") as f:
    for t in hard + extra:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
print(f"{len(hard)} hard + {len(extra)} parallelisable -> {a.out}")
