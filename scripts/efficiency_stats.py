"""Results table for the efficiency experiments (call style, parallel calls, efficiency reward).

Per split: success, turns, tool calls, generated tokens and false completions for each run,
plus paired McNemar tests between the runs that matter.

    python scripts/efficiency_stats.py          # writes results/efficiency.md and .json
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stats import mcnemar_exact, paired_bootstrap  # noqa: E402

SPLITS = ["test_id", "test_expr", "test_entity", "test_combo"]
RUNS = {
    "grpo_v2": "JSON, one call per turn, SFT -> GRPO (main result)",
    "sft_fn_par": "compact fn style, 30% parallel SFT",
    "grpo_fn_base": "  -> GRPO",
    "grpo_fn_eff": "  -> GRPO + turn penalty (applied to every episode)",
    "sft_js_par": "JSON, 30% parallel SFT",
    "grpo_js_base": "  -> GRPO",
    "grpo_js_eff": "  -> GRPO + turn penalty (successful episodes only)",
}
PAIRS = [("grpo_js_eff", "grpo_js_base"), ("grpo_js_base", "grpo_v2"), ("grpo_js_base", "sft_js_par"),
         ("grpo_fn_base", "grpo_v2")]


def load(run, split):
    for name in (run, run + "_combo", "dpo_v1_combo" if run == "dpo_v1" else run):
        p = Path("runs/eval") / name / f"episodes_{split}.jsonl"
        if p.exists():
            return {json.loads(l)["task_id"]: json.loads(l)["verify"] for l in open(p, encoding="utf-8")}
    return None


def main():
    data = {r: {s: load(r, s) for s in SPLITS} for r in RUNS}
    out, res = ["# Efficiency experiments (300 tasks per split, same tasks for every run, one seed)", ""], {}
    for s in SPLITS:
        out += [f"## {s}", "", "| Run | Setup | Success | Turns | Tool calls | Generated tokens | False completion |",
                "|---|---|---:|---:|---:|---:|---:|"]
        for r, desc in RUNS.items():
            v = data[r][s]
            if not v:
                continue
            vv = list(v.values())
            row = {"success": st.fmean(x["success"] for x in vv), "turns": st.fmean(x["turns"] for x in vv),
                   "calls": st.fmean(x["n_calls"] for x in vv), "false_done": st.fmean(x.get("false_done", 0) for x in vv),
                   "tokens": st.fmean(x["gen_tokens"] for x in vv) if "gen_tokens" in vv[0] else None}
            res.setdefault(r, {})[s] = row
            tok = f"{row['tokens']:.1f}" if row["tokens"] is not None else "—"
            out.append(f"| {r} | {desc} | {100 * row['success']:.1f}% | {row['turns']:.2f} | {row['calls']:.2f} | "
                       f"{tok} | {100 * row['false_done']:.1f}% |")
        out += ["", "| A vs B | Success diff | 95% bootstrap CI | only A | only B | McNemar p | Turns A / B |",
                "|---|---:|---:|---:|---:|---:|---:|"]
        for a, b in PAIRS:
            va, vb = data[a][s], data[b][s]
            ids = sorted(set(va) & set(vb))
            xa = [int(va[i]["success"]) for i in ids]
            xb = [int(vb[i]["success"]) for i in ids]
            g = sum(1 for p, q in zip(xa, xb) if p and not q)
            l = sum(1 for p, q in zip(xa, xb) if q and not p)
            lo, hi = paired_bootstrap(xa, xb)
            ta = st.fmean(va[i]["turns"] for i in ids)
            tb = st.fmean(vb[i]["turns"] for i in ids)
            out.append(f"| {a} vs {b} | {100 * (sum(xa) - sum(xb)) / len(ids):+.1f} | [{100 * lo:+.1f}, {100 * hi:+.1f}] "
                       f"| {g} | {l} | {mcnemar_exact(g, l):.3g} | {ta:.2f} / {tb:.2f} ({100 * (ta / tb - 1):+.0f}%) |")
        out.append("")
    Path("results").mkdir(exist_ok=True)
    Path("results/efficiency.md").write_text("\n".join(out), encoding="utf-8")
    Path("results/efficiency.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
