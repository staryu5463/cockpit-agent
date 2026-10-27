"""Episode runner shared by every policy (expert, rules, HF model, vLLM)."""
from __future__ import annotations

import json
import time

from cockpit import spec
from cockpit.agent import format as F
from cockpit.agent.format import build_system_prompt, parse_tool_calls
from cockpit.env.episode import Episode
from cockpit.env.tools import MANUAL_TOOL_SPEC, TOOL_SPECS
from cockpit.verifier import verify


def initial_messages(task, enable_manual=False):
    tools = TOOL_SPECS + ([MANUAL_TOOL_SPEC] if enable_manual else [])
    return [{"role": "system", "content": build_system_prompt(tools)},
            {"role": "user", "content": task["text"]}]


def _format_error(msg):
    return {"ok": False, "code": "FORMAT_ERROR", "data": msg, "retryable": True}


def execute_turn(ep, text, parallel=None):
    """Run the tool calls of one assistant turn -> (tool message content, had_format_error).

    Single-call mode keeps the first call only (the original behaviour). Parallel mode
    runs up to MAX_CALLS_PER_TURN calls in order and returns their results as a list;
    a turn that mixes finish with other calls is rejected unexecuted, so the agent can
    never report "done" before it has seen the results of its own actions.
    """
    parallel = F.PARALLEL if parallel is None else parallel
    calls = parse_tool_calls(text)
    if not calls:
        return json.dumps(_format_error("no <tool_call> block"), ensure_ascii=False), True
    if not parallel:
        calls = calls[:1]
    elif len(calls) > F.MAX_CALLS_PER_TURN:
        return json.dumps(_format_error(f"at most {F.MAX_CALLS_PER_TURN} calls per turn"), ensure_ascii=False), True
    elif len(calls) > 1 and any(n == "finish" for n, _, _ in calls):
        return json.dumps(_format_error("finish must be called alone"), ensure_ascii=False), True
    results, bad = [], False
    for name, args, perr in calls:
        if perr:
            results.append(_format_error(perr))
            bad = True
        else:
            results.append(ep.call(name, args))
    content = results[0] if len(results) == 1 else results
    return json.dumps(content, ensure_ascii=False), bad


def finalize(ep, messages, format_errors):
    v = verify(ep)
    v["format_errors"] = format_errors
    assistant = [m for m in messages if m["role"] == "assistant"]
    v["turns"] = len(assistant)
    v["gen_tokens"] = sum(len(m["token_ids"]) for m in assistant if m.get("token_ids"))
    return v


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
        content, bad = execute_turn(ep, out["text"])
        format_errors += bad
        messages.append({"role": "tool", "content": content})
        if ep.finished or ep.budget_left <= 0:
            break
    return {"task_id": task["id"], "tier": task["tier"], "messages": messages,
            "verify": finalize(ep, messages, format_errors), "seconds": time.perf_counter() - t0, "episode": ep}
