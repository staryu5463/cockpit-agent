"""Reward = verifiable outcome + small, separately ablatable shaping terms.

    R = w_success * success
      + w_format  * (no format errors)
      - w_steps   * max(0, calls - expert_calls)
      - w_side    * (#side-effect fields)
      - w_turns   * max(0, turns - expert_turns)        (efficiency; 0 unless set)
      - w_tokens  * generated tokens                     (efficiency; 0 unless set)

Principle: the main term only looks at the final state and the finish status
(checked by code). Shaping terms are small and each has an on/off switch, so
an ablation can show whether it helps or gets exploited.

eff_on_success=1 applies the two efficiency terms to successful episodes only. Without
it the turn penalty was exploited: the policy skipped retries and rushed choices, trading
success for fewer turns (combination success 85% -> 72% in the first run).
"""
from __future__ import annotations

DEFAULT_WEIGHTS = {"success": 1.0, "format": 0.1, "steps": 0.05, "side": 0.2, "turns": 0.0, "tokens": 0.0,
                   "eff_on_success": 0}


def compute_reward(v: dict, expert_calls: int | None, w: dict = None, expert_turns: int | None = None) -> float:
    w = {**DEFAULT_WEIGHTS, **(w or {})}
    r = w["success"] * float(v["success"])
    r += w["format"] * float(v.get("format_errors", 0) == 0)
    if expert_calls is not None:
        r -= w["steps"] * max(0, v["n_calls"] - expert_calls)
    r -= w["side"] * len(v["side_effects"])
    if v["success"] or not w["eff_on_success"]:
        if expert_turns is not None:
            r -= w["turns"] * max(0, v.get("turns", 0) - expert_turns)
        r -= w["tokens"] * v.get("gen_tokens", 0)
    return r
