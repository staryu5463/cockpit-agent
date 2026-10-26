"""Evaluation harness: same tasks, same tools, same budget for every method.

    python -m cockpit.eval.evaluate --policy rule   --splits dev test_expr
    python -m cockpit.eval.evaluate --policy hf     --model <path> --lora runs/sft/lora.pt
    python -m cockpit.eval.evaluate --policy vllm   --base-url http://localhost:8000/v1 --model <name>

Writes runs/eval/<name>/summary.json, a markdown table, and every episode
(for failure analysis) to episodes_<split>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from cockpit.agent.runner import run_episode
from cockpit.tasks.splits import load_split

SPLITS = ["dev", "test_id", "test_expr", "test_entity", "test_combo"]


def summarize(results):
    def agg(rs):
        if not rs:
            return {}
        v = [r["verify"] for r in rs]
        return {
            "n": len(rs),
            "success": round(statistics.fmean(x["success"] for x in v), 4),
            "state_ok": round(statistics.fmean(x["state_ok"] for x in v), 4),
            "finish_ok": round(statistics.fmean(x["finish_ok"] for x in v), 4),
            "side_effect_rate": round(statistics.fmean(bool(x["side_effects"]) for x in v), 4),
            "false_done_rate": round(statistics.fmean(x.get("false_done", False) for x in v), 4),
            "format_error_rate": round(statistics.fmean(x.get("format_errors", 0) > 0 for x in v), 4),
            "failed_calls_per_ep": round(statistics.fmean(x["n_failed_calls"] for x in v), 3),
            "calls_per_ep": round(statistics.fmean(x["n_calls"] for x in v), 3),
            "sec_per_ep": round(statistics.fmean(r["seconds"] for r in rs), 4),
        }
    by_tier = defaultdict(list)
    for r in results:
        by_tier[r["tier"]].append(r)
    return {"overall": agg(results), "by_tier": {k: agg(v) for k, v in sorted(by_tier.items())}}


def make_policy(a):
    if a.policy == "expert":
        from cockpit.baselines.planner import ExpertPolicy
        return ExpertPolicy()
    if a.policy == "rule":
        from cockpit.baselines.rule_nlu import RulePolicy
        return RulePolicy()
    if a.policy == "hf":
        import torch
        from cockpit.agent.format import ChatFormat
        from cockpit.train.common import HFPolicy, assistant_prefix_for, load_model
        from cockpit.train.lora import apply_lora, load_lora_state_dict
        model, tok = load_model(a.model, device=a.device)
        if a.lora:
            apply_lora(model, r=a.lora_r, alpha=a.lora_alpha, dropout=0.0)
            load_lora_state_dict(model, torch.load(a.lora, map_location="cpu"))
        fmt = ChatFormat(tok, assistant_prefix_for(a.model))
        return HFPolicy(model.eval(), fmt, greedy=True, max_new_tokens=a.max_new_tokens)
    if a.policy == "vllm":
        from cockpit.serve.client import VLLMPolicy
        return VLLMPolicy(a.base_url, a.model, tokenizer_path=a.tokenizer or a.model,
                          guided=a.guided, max_new_tokens=a.max_new_tokens)
    raise ValueError(a.policy)


def strip(res):
    msgs = [{k: v for k, v in m.items() if k != "token_ids"} for m in res["messages"][1:]]
    return {"task_id": res["task_id"], "tier": res["tier"], "verify": res["verify"],
            "seconds": res["seconds"], "messages": msgs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=["expert", "rule", "hf", "vllm"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--model"); ap.add_argument("--lora"); ap.add_argument("--tokenizer")
    ap.add_argument("--lora-r", type=int, default=16); ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--guided", action="store_true", help="vLLM JSON-constrained decoding")
    ap.add_argument("--enable-manual", action="store_true")
    ap.add_argument("--hybrid", action="store_true", help="route simple commands to the rule fast path")
    ap.add_argument("--batch-size", type=int, default=1, help="batched rollouts for --policy hf")
    a = ap.parse_args()

    policy = make_policy(a)
    if a.hybrid:
        from cockpit.serve.router import HybridPolicy
        policy = HybridPolicy(policy)
    name = a.name or a.policy
    out = Path("runs/eval") / name
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for sp in a.splits:
        tasks = load_split(Path(a.data_dir) / f"{sp}.jsonl")[: a.limit]
        if a.batch_size > 1 and hasattr(policy, "act_batch"):
            from cockpit.agent.batch_runner import run_episodes_batch
            results = []
            for i in range(0, len(tasks), a.batch_size):
                results += run_episodes_batch(tasks[i:i + a.batch_size], policy, enable_manual=a.enable_manual)
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()     # avoid spilling into shared memory on 8 GB cards
        else:
            results = [run_episode(t, policy, enable_manual=a.enable_manual) for t in tasks]
        summary[sp] = summarize(results)
        with open(out / f"episodes_{sp}.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(strip(r), ensure_ascii=False) + "\n")
        o = summary[sp]["overall"]
        print(f"{sp:12s} n={o['n']:4d} success={o['success']:.3f} side_effect={o['side_effect_rate']:.3f} "
              f"format_err={o['format_error_rate']:.3f} false_done={o['false_done_rate']:.3f} calls={o['calls_per_ep']:.2f}")
    if a.hybrid:
        summary["routing"] = policy.routed
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    with open(out / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# {name}\n\n| split | n | success | side-effect | format-err | calls/ep |\n|---|---:|---:|---:|---:|---:|\n")
        for sp, s in summary.items():
            if sp == "routing":
                continue
            o = s["overall"]
            f.write(f"| {sp} | {o['n']} | {o['success']:.3f} | {o['side_effect_rate']:.3f} | "
                    f"{o['format_error_rate']:.3f} | {o['calls_per_ep']:.2f} |\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
