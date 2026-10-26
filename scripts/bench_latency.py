"""Latency benchmark against a running vLLM server.

Runs the same N episodes and reports per-request and per-episode latency,
plus success, so speed is never reported without quality.

    python scripts/bench_latency.py --split test_id --n 100 --tag prefix
    python scripts/bench_latency.py --split test_id --n 100 --tag prefix_guided --guided
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cockpit.agent.runner import run_episode  # noqa: E402
from cockpit.serve.client import VLLMPolicy  # noqa: E402
from cockpit.tasks.splits import load_split  # noqa: E402


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://localhost:8000/v1")
ap.add_argument("--model", default="cockpit")
ap.add_argument("--tokenizer", default="qwen3")
ap.add_argument("--split", default="test_id")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--guided", action="store_true")
ap.add_argument("--tag", required=True)
a = ap.parse_args()

pol = VLLMPolicy(a.base_url, a.model, tokenizer_path=a.tokenizer, guided=a.guided)
tasks = load_split(f"data/{a.split}.jsonl")[: a.n]
run_episode(tasks[0], pol)                                  # warm-up
req, eps, ok = [], [], []
for t in tasks:
    r = run_episode(t, pol)
    req += pol.last_latency
    eps.append(r["seconds"])
    ok.append(r["verify"]["success"])
res = {"tag": a.tag, "n": len(tasks), "success": statistics.fmean(ok),
       "req_p50_ms": 1000 * pct(req, .5), "req_p95_ms": 1000 * pct(req, .95),
       "ep_p50_ms": 1000 * pct(eps, .5), "ep_p95_ms": 1000 * pct(eps, .95),
       "calls_per_ep": len(req) / len(tasks)}
print(json.dumps(res, indent=2))
Path("runs/bench").mkdir(parents=True, exist_ok=True)
with open("runs/bench/latency.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(res) + "\n")
