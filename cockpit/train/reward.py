"""Reward = verifiable outcome + small, separately ablatable shaping terms.

    R = w_success * success
      + w_format  * (no format errors)
      - w_steps   * max(0, calls - expert_calls)
      - w_side    * (#side-effect fields)

Principle: the main term only looks at the final state and the finish status
(checked by code). Shaping terms are small and each has an on/off switch, so
an ablation can show whether it helps or gets exploited.
"""
from __future__ import annotations

DEFAULT_WEIGHTS = {"success": 1.0, "format": 0.1, "steps": 0.05, "side": 0.2}


def compute_reward(v: dict, expert_calls: int | None, w: dict = None) -> float:
    w = {**DEFAULT_WEIGHTS, **(w or {})}
    r = w["success"] * float(v["success"])
    r += w["format"] * float(v.get("format_errors", 0) == 0)
    if expert_calls is not None:
        r -= w["steps"] * max(0, v["n_calls"] - expert_calls)
    r -= w["side"] * len(v["side_effects"])
    return r
