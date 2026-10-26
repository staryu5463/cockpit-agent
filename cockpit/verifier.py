"""Private verifier: scores a finished episode from state + trajectory.

The model saying "done" never decides the score. A task succeeds only if
  1. every goal field has an acceptable final value,
  2. no other field changed (no side effects),
  3. the episode ended with the expected finish status.
"""
from __future__ import annotations

from cockpit.env import state as st


def _field_ok(expected, actual) -> bool:
    if isinstance(expected, dict) and "any_of" in expected:
        return actual in expected["any_of"]
    return actual == expected


def verify(ep) -> dict:
    goal = ep.task["goal"]
    goal_fields = goal.get("fields", {})
    final = ep.state
    changed = st.diff(ep.initial_state, final)

    wrong_fields = [f for f, exp in goal_fields.items() if not _field_ok(exp, final[f])]
    side_effects = sorted(f for f in changed if f not in goal_fields)
    finish_status = ep.finish_payload["status"] if ep.finish_payload else None
    finish_ok = finish_status == goal["finish_status"]

    failed_calls = [c for c in ep.calls if not c["result"]["ok"]]
    recoverable = [c for c in failed_calls if c["result"]["retryable"]]
    read_status = any(c["name"] == "get_vehicle_status" and c["result"]["ok"] for c in ep.calls)

    cite_ok = True
    if goal.get("must_cite"):
        # must have retrieved the passage AND cite its id in the answer
        msg = (ep.finish_payload or {}).get("message", "")
        cite_ok = goal["must_cite"] in ep.surfaced and goal["must_cite"] in msg
    success = not wrong_fields and not side_effects and finish_ok and cite_ok
    # "hallucinated completion": the agent told the user it was done, but it was not
    false_done = finish_status == "done" and not success
    reasons = []
    if wrong_fields:
        reasons.append("wrong:" + ",".join(wrong_fields))
    if side_effects:
        reasons.append("side_effect:" + ",".join(side_effects))
    if not cite_ok:
        reasons.append("citation")
    if not ep.finished:
        reasons.append("no_finish")
    elif not finish_ok:
        reasons.append(f"finish:{finish_status}!={goal['finish_status']}")

    return {
        "success": success,
        "state_ok": not wrong_fields,
        "finish_ok": finish_ok,
        "cite_ok": cite_ok,
        "false_done": false_done,
        "finished": ep.finished,
        "wrong_fields": wrong_fields,
        "side_effects": side_effects,
        "n_calls": len(ep.calls),
        "n_failed_calls": len(failed_calls) - len(recoverable),  # agent mistakes only
        "read_status": read_status,
        "reasons": reasons,
    }
