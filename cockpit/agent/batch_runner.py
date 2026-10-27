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

import time

from cockpit import spec
from cockpit.agent.runner import execute_turn, finalize, initial_messages, run_episode
from cockpit.env.episode import Episode


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
        active = [i for i, e in enumerate(eps) if not e.finished and e.budget_left > 0]
        if not active:
            break
        outs = policy.act_batch([msgs[i] for i in active])
        for i, out in zip(active, outs):
            msgs[i].append({"role": "assistant", "content": out["text"], "token_ids": out.get("token_ids")})
            content, bad = execute_turn(eps[i], out["text"])
            ferr[i] += bad
            msgs[i].append({"role": "tool", "content": content})
    dt = (time.perf_counter() - t0) / max(1, len(tasks))
    results = []
    for t, ep, m, fe in zip(tasks, eps, msgs, ferr):
        results.append({"task_id": t["id"], "tier": t["tier"], "messages": m, "verify": finalize(ep, m, fe),
                        "seconds": dt, "episode": ep})
    return results
