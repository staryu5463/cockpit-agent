"""Export agent conversations for training a speculative-decoding draft head.

Hypothesis to test: tool calls are highly structured JSON, so a draft head
trained on in-domain agent trajectories should be accepted more often than a
general-purpose one. This script writes ShareGPT-style conversations that
EAGLE-3 training code (e.g. the official EAGLE repo or SpecForge) can consume;
check the exact input format of the trainer version you use.

    python scripts/export_draft_data.py --split train --out data/draft_train.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cockpit.agent.runner import run_episode  # noqa: E402
from cockpit.baselines.planner import ExpertPolicy  # noqa: E402
from cockpit.tasks.splits import load_split  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="train")
ap.add_argument("--out", required=True)
a = ap.parse_args()

ROLE = {"system": "system", "user": "human", "assistant": "gpt", "tool": "human"}
n = 0
with open(a.out, "w", encoding="utf-8") as f:
    for t in load_split(f"data/{a.split}.jsonl"):
        r = run_episode(t, ExpertPolicy())
        conv = []
        for m in r["messages"]:
            text = m["content"] if m["role"] != "tool" else f"<tool_response>\n{m['content']}\n</tool_response>"
            conv.append({"from": ROLE[m["role"]], "value": text})
        f.write(json.dumps({"id": t["id"], "conversations": conv}, ensure_ascii=False) + "\n")
        n += 1
print("wrote", n, "conversations to", a.out)
