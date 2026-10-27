"""Trajectory failure analysis: attribute every failed episode to a root cause.

Reads runs/eval/<name>/episodes_<split>.jsonl and walks each trajectory.
Causes are checked in a fixed priority order, so every failure gets exactly
one primary label (plus all secondary labels for the detailed view):

  format          a turn had no parsable <tool_call>
  hallucinated_id used an id that no search returned (UNKNOWN_ID)
  skipped_read    relative task, wrote state without reading it first
  bad_args        INVALID_ARGUMENT / OUT_OF_RANGE on a write
  no_retry        a retryable failure was never retried
  wrong_choice    searched, but picked a closed / not-nearest / wrong item
  misunderstood   no tool error at all, just the wrong target values (intent / slot error)
  side_effect     changed settings nobody asked for
  wrong_status    ended with the wrong finish status (e.g. executed instead of asking)
  no_finish       ran out of turns / budget without finishing
  other

    python scripts/analyze_failures.py runs/eval/sft_0.6b --split test_expr
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cockpit.tasks.splits import load_split  # noqa: E402

WRITE_TOOLS = {"ac_power", "ac_set_temperature", "volume_set", "nav_start", "media_play"}
ORDER = ["format", "hallucinated_id", "skipped_read", "bad_args", "no_retry", "wrong_choice",
         "misunderstood", "side_effect", "wrong_status", "no_finish", "other"]


def calls(ep):
    """[(name, args, result)] from the logged messages."""
    from cockpit.agent.format import parse_tool_calls
    out, pending = [], None
    for m in ep["messages"]:
        if m["role"] == "assistant":
            parsed = parse_tool_calls(m["content"]) or [(None, None, "no call")]
            pending = [(n, a) if not err else ("<format>", None) for n, a, err in parsed]
        elif m["role"] == "tool" and pending:
            res = json.loads(m["content"])
            res = res if isinstance(res, list) else [res]          # parallel turns return a list
            if len(res) != len(pending):                          # single-call mode or rejected turn
                pending, res = pending[:1], res[:1]
            out += [(n, a, r) for (n, a), r in zip(pending, res)]
            pending = None
    return out


def causes(ep, task):
    v = ep["verify"]
    cs = calls(ep)
    labels = set()
    if v.get("format_errors", 0) > 0:
        labels.add("format")
    if any(r.get("code") == "UNKNOWN_ID" for _, _, r in cs):
        labels.add("hallucinated_id")
    rel_tools = {"TEMP_REL": "ac_set_temperature", "VOL_REL": "volume_set"}
    rel = {rel_tools[op["op"]] for op in task["semantic"] if op["op"] in rel_tools}
    if rel:
        # only the write that applies the relative change must come after a state read
        first_write = next((i for i, (n, _, _) in enumerate(cs) if n in rel), None)
        first_read = next((i for i, (n, _, r) in enumerate(cs) if n == "get_vehicle_status" and r.get("ok")), None)
        if first_write is not None and (first_read is None or first_read > first_write):
            labels.add("skipped_read")
    if any(n in WRITE_TOOLS and r.get("code") in ("INVALID_ARGUMENT", "OUT_OF_RANGE") for n, _, r in cs):
        labels.add("bad_args")
    for i, (n, a, r) in enumerate(cs):
        if r.get("retryable") and r.get("code") == "SERVICE_BUSY":
            if not any(n2 == n for n2, _, _ in cs[i + 1:]):
                labels.add("no_retry")
    searched = any(n in ("search_poi", "media_search") for n, _, _ in cs)
    if searched and any(w.startswith("wrong:") and ("nav." in w or "media.playing" in w) for w in v["reasons"]):
        labels.add("wrong_choice")
    if v["wrong_fields"] and not labels:
        labels.add("misunderstood")
    if v["side_effects"]:
        labels.add("side_effect")
    if v["finished"] and not v["finish_ok"]:
        labels.add("wrong_status")
    if not v["finished"]:
        labels.add("no_finish")
    if not labels:
        labels.add("other")
    primary = next(c for c in ORDER if c in labels)
    return primary, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--examples", type=int, default=2)
    a = ap.parse_args()
    tasks = {t["id"]: t for t in load_split(Path(a.data_dir) / f"{a.split}.jsonl")}
    eps = [json.loads(l) for l in open(Path(a.eval_dir) / f"episodes_{a.split}.jsonl", encoding="utf-8")]
    fails = [e for e in eps if not e["verify"]["success"]]
    prim, anyc, by_tier, ex = Counter(), Counter(), defaultdict(Counter), defaultdict(list)
    for e in fails:
        p, ls = causes(e, tasks[e["task_id"]])
        prim[p] += 1
        anyc.update(ls)
        by_tier[e["tier"]][p] += 1
        if len(ex[p]) < a.examples:
            ex[p].append(tasks[e["task_id"]]["text"] + "  ->  " + ",".join(e["verify"]["reasons"]))
    n = len(eps)
    def said_done(e):      # read from the trajectory itself, works for older eval runs too
        cs = calls(e)
        return any(n == "finish" and isinstance(a, dict) and a.get("status") == "done" and r.get("ok")
                   for n, a, r in cs)
    false_done = sum(said_done(e) and not e["verify"]["success"] for e in eps)
    print(f"{a.split}: {len(fails)}/{n} failed; hallucinated completion (said done, was not) {false_done}/{n}\n")
    print(f"{'cause':16s} {'primary':>8s} {'any':>6s}")
    for c in ORDER:
        if anyc[c]:
            print(f"{c:16s} {prim[c]:8d} {anyc[c]:6d}")
    print("\nby tier (primary cause):")
    for t, c in sorted(by_tier.items()):
        print(f"  {t:12s} " + ", ".join(f"{k}={v}" for k, v in c.most_common()))
    print("\nexamples:")
    for c in ORDER:
        for s in ex.get(c, []):
            print(f"  [{c}] {s}")
    out = Path(a.eval_dir) / f"failures_{a.split}.json"
    json.dump({"primary": prim, "any": anyc, "by_tier": by_tier}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
