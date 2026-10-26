"""Data flywheel: turn evaluation failures into new training tasks.

1. Read episodes from an evaluation run (on train or dev only, never test).
2. Count which ops / tiers the model fails on.
3. Generate fresh tasks with the same ops but new seeds (new states, IDs,
   wording), so the model gets more of what it gets wrong without seeing
   the failed sentences again.

    python scripts/flywheel.py --eval runs/eval/sft_dev --data data --n 500 --out data/flywheel_r1.jsonl
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cockpit.tasks.generator import TRAIN_FAMILIES, make_task  # noqa: E402
from cockpit.tasks.splits import _no_heldout_combo, load_split  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--eval", required=True, help="runs/eval/<name> directory")
ap.add_argument("--split", default="dev", choices=["train", "dev"])
ap.add_argument("--data", default="data")
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=123)
a = ap.parse_args()

tasks = {t["id"]: t for t in load_split(Path(a.data) / f"{a.split}.jsonl")}
fail_tiers = Counter()
reasons = Counter()
for line in open(Path(a.eval) / f"episodes_{a.split}.jsonl", encoding="utf-8"):
    e = json.loads(line)
    if not e["verify"]["success"]:
        fail_tiers[tasks[e["task_id"]]["tier"]] += 1
        reasons.update(r.split(":")[0] for r in e["verify"]["reasons"])
print("failures by tier:", dict(fail_tiers))
print("failure reasons:", dict(reasons))
if not fail_tiers:
    sys.exit("no failures to learn from")

rng = random.Random(a.seed)
tiers, weights = zip(*fail_tiers.items())
with open(a.out, "w", encoding="utf-8") as f:
    for i in range(a.n):
        tier = rng.choices(tiers, weights=weights)[0]
        t = make_task(rng, tier, rng.choice(TRAIN_FAMILIES), allow_combo=_no_heldout_combo,
                      task_id=f"fly-{a.seed}-{i:05d}")
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
print("wrote", a.out)
