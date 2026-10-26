"""Optional LLM paraphrasing (any OpenAI-compatible API, e.g. DeepSeek).

The goal was computed by code from the semantic task, so a paraphrase may only
change wording. `consistent()` rejects rewrites that drop or change a number,
a place/brand, a song or an artist. Rejected rewrites are logged, not used.

    export LLM_BASE_URL=https://api.deepseek.com/v1  LLM_API_KEY=...  LLM_MODEL=deepseek-chat
    python -m cockpit.tasks.paraphrase data/train.jsonl data/train_para.jsonl --per-task 2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request

PROMPT = """把下面这句车载语音指令改写成 {k} 种口语化的说法，意思、数字、地点名、歌名、歌手名必须完全保留，
不要增加或删减任何操作。每行一句，不要编号。
原句：{text}"""


def _must_keep(task):
    keep = set(re.findall(r"\d+", task["text"]))
    for op in task["semantic"]:
        for k in ("brand", "title", "artist"):
            if k in op and op[k] in task["text"]:
                keep.add(op[k])
    return keep


def consistent(task, new_text):
    return all(x in new_text for x in _must_keep(task)) and new_text != task["text"]


def chat(prompt):
    body = {"model": os.environ.get("LLM_MODEL", "deepseek-chat"), "temperature": 0.9,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {os.environ['LLM_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--per-task", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    from cockpit.tasks.splits import load_split
    tasks = load_split(a.src)[: a.limit]
    kept = rejected = 0
    with open(a.dst, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
            if t["tier"] == "chat":
                continue
            lines = [l.strip() for l in chat(PROMPT.format(k=a.per_task, text=t["text"])).splitlines() if l.strip()]
            for j, line in enumerate(lines[: a.per_task]):
                if consistent(t, line):
                    v = dict(t, id=f"{t['id']}-p{j}", text=line, family=t["family"] + "-para")
                    f.write(json.dumps(v, ensure_ascii=False) + "\n"); kept += 1
                else:
                    rejected += 1
    print(f"kept {kept} paraphrases, rejected {rejected}")


if __name__ == "__main__":
    main()
