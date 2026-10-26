"""Uncertainty for the results table: 95% intervals and paired significance tests.

Every method was evaluated on the same 300 tasks per split, so differences are
tested *paired*: McNemar's exact test on the tasks where the two methods disagree,
plus a paired bootstrap interval for the difference in success rate.

    python scripts/stats.py                       # writes runs/eval/stats.md and stats.json
    python scripts/stats.py --metric false_done
"""
import argparse
import json
import math
import random
from pathlib import Path

SPLITS = ["test_id", "test_expr", "test_entity", "test_combo"]
METHODS = {"rule": "rule_full", "sft": "sft_0.6b", "dpo": "dpo_v1", "grpo": "grpo_v2"}
OVERRIDE = {("dpo", "test_combo"): "dpo_v1_combo"}         # combo was re-run separately
PAIRS = [("grpo", "sft"), ("grpo", "dpo"), ("grpo", "rule"), ("dpo", "sft"), ("sft", "rule")]


def load(method, split, metric):
    run = OVERRIDE.get((method, split), METHODS[method])
    path = Path("runs/eval") / run / f"episodes_{split}.jsonl"
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out[r["task_id"]] = int(bool(r["verify"][metric]))
    return out


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(b, c):
    """Two-sided exact binomial test on the discordant pairs (b: only A right, c: only B right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_bootstrap(a, b, iters=10000, seed=0):
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="success")
    a = ap.parse_args()
    data = {m: {s: load(m, s, a.metric) for s in SPLITS} for m in METHODS}
    lines = [f"# Uncertainty for `{a.metric}` (300 tasks per split, same tasks for every method)", "",
             "## Rates with 95% Wilson intervals", "",
             "| Method | " + " | ".join(SPLITS) + " |", "|---|" + "---:|" * len(SPLITS)]
    res = {"metric": a.metric, "rates": {}, "pairs": []}
    for m in METHODS:
        cells = []
        for s in SPLITS:
            v = list(data[m][s].values())
            k, n = sum(v), len(v)
            lo, hi = wilson(k, n)
            res["rates"].setdefault(m, {})[s] = {"rate": k / n, "lo": lo, "hi": hi, "n": n}
            cells.append(f"{100 * k / n:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired differences (A − B), percentage points", "",
              "| A vs B | split | diff | 95% bootstrap CI | only A | only B | McNemar p |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for x, y in PAIRS:
        for s in SPLITS:
            ids = sorted(set(data[x][s]) & set(data[y][s]))
            va = [data[x][s][i] for i in ids]
            vb = [data[y][s][i] for i in ids]
            b = sum(1 for p, q in zip(va, vb) if p and not q)
            c = sum(1 for p, q in zip(va, vb) if q and not p)
            diff = (sum(va) - sum(vb)) / len(ids)
            lo, hi = paired_bootstrap(va, vb)
            p = mcnemar_exact(b, c)
            res["pairs"].append({"a": x, "b": y, "split": s, "n": len(ids), "diff": diff, "lo": lo, "hi": hi,
                                 "only_a": b, "only_b": c, "p": p})
            lines.append(f"| {x} vs {y} | {s} | {100 * diff:+.1f} | [{100 * lo:+.1f}, {100 * hi:+.1f}] "
                         f"| {b} | {c} | {p:.3g} |")
    suffix = "" if a.metric == "success" else f"_{a.metric}"
    Path(f"runs/eval/stats{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(f"runs/eval/stats{suffix}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
