"""Episode runner shared by every policy (expert, rules, HF model, vLLM)."""
from __future__ import annotations

import json
import time

from cockpit import spec
from cockpit.agent.format import build_system_prompt, parse_tool_call
from cockpit.env.episode import Episode
from cockpit.env.tools import MANUAL_TOOL_SPEC, TOOL_SPECS
from cockpit.verifier import verify


def initial_messages(task, enable_manual=False):
    tools = TOOL_SPECS + ([MANUAL_TOOL_SPEC] if enable_manual else [])
    return [{"role": "system", "content": build_system_prompt(tools)},
            {"role": "user", "content": task["text"]}]


def run_episode(task: dict, policy, *, enable_manual=False, max_turns=spec.MAX_TOOL_CALLS) -> dict:
    """policy.reset(task); policy.act(messages) -> {"text": str, "token_ids": list|None}."""
    enable_manual = enable_manual or task.get("enable_manual", False)
    ep = Episode(task, enable_manual=enable_manual)
    messages = initial_messages(task, enable_manual)
    policy.reset(task)
    format_errors = 0
    t0 = time.perf_counter()
    for _ in range(max_turns):
        out = policy.act(messages)
        messages.append({"role": "assistant", "content": out["text"],
                         "token_ids": out.get("token_ids")})
        name, args, perr = parse_tool_call(out["text"])
        if perr:
            format_errors += 1
            result = {"ok": False, "code": "FORMAT_ERROR", "data": perr, "retryable": True}
        else:
            result = ep.call(name, args)
        messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
        if ep.finished:
            break
    v = verify(ep)
    v["format_errors"] = format_errors
    v["turns"] = sum(1 for m in messages if m["role"] == "assistant")
    return {"task_id": task["id"], "tier": task["tier"], "messages": messages,
            "verify": v, "seconds": time.perf_counter() - t0, "episode": ep}
