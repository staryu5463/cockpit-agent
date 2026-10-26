"""Generate all splits into data/.

    python scripts/make_data.py                 # default sizes
    python scripts/make_data.py --small         # 200 train / 50 per test, for a first look
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cockpit.tasks.splits import build_all  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--small", action="store_true")
a = ap.parse_args()
sizes = ({"train": 200, "dev": 50, "test_id": 50, "test_expr": 50, "test_entity": 50, "test_combo": 50,
          "train_manual": 40, "test_manual": 20}
         if a.small else None)
print(build_all(a.out, sizes, a.seed))
