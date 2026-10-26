"""Batched multi-turn rollouts.

`run_episode` drives one conversation at a time, so a GRPO group of 8 makes
8 x (number of turns) separate `generate` calls. Here the episodes advance in
lock-step: at every turn, all unfinished episodes are generated in ONE batched
call (left-padded), then each one executes its own tool call in its own
environment. Same results structure as `run_episode`, many fewer GPU calls.

Only stateless model policies (HFPolicy) support `act_batch`; scripted
policies keep per-episode state, so they fall back to `run_episode`.
"""
from __future__ import annotations

import json
import time

from cockpit import spec
from cockpit.agent.format import parse_tool_call
from cockpit.agent.runner import initial_messages, run_episode
from cockpit.env.episode import Episode
from cockpit.verifier import verify


def run_episodes_batch(tasks, policy, *, enable_manual=False, max_turns=spec.MAX_TOOL_CALLS):
    if not hasattr(policy, "act_batch"):
        return [run_episode(t, policy, enable_manual=enable_manual, max_turns=max_turns) for t in tasks]
    t0 = time.perf_counter()
    eps, msgs, ferr = [], [], []
    for t in tasks:
        em = enable_manual or t.get("enable_manual", False)
        eps.append(Episode(t, enable_manual=em))
        msgs.append(initial_messages(t, em))
        ferr.append(0)
    for _ in range(max_turns):
        active = [i for i, e in enumerate(eps) if not e.finished]
        if not active:
            break
        outs = policy.act_batch([msgs[i] for i in active])
        for i, out in zip(active, outs):
            msgs[i].append({"role": "assistant", "content": out["text"], "token_ids": out.get("token_ids")})
            name, args, perr = parse_tool_call(out["text"])
            if perr:
                ferr[i] += 1
                result = {"ok": False, "code": "FORMAT_ERROR", "data": perr, "retryable": True}
            else:
                result = eps[i].call(name, args)
            msgs[i].append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
    dt = (time.perf_counter() - t0) / max(1, len(tasks))
    results = []
    for t, ep, m, fe in zip(tasks, eps, msgs, ferr):
        v = verify(ep)
        v["format_errors"] = fe
        v["turns"] = sum(1 for x in m if x["role"] == "assistant")
        results.append({"task_id": t["id"], "tier": t["tier"], "messages": m, "verify": v,
                        "seconds": dt, "episode": ep})
    return results
