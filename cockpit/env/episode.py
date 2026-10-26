"""One episode = one user request handled against one simulated cockpit.

The policy only ever talks to `Episode.call`. The goal, the catalog internals
and the verifier stay private.
"""
from __future__ import annotations

import inspect
import random

from cockpit import spec
from cockpit.env import state as st
from cockpit.env.catalog import make_catalog
from cockpit.env.tools import IMPLS, CORE_TOOL_NAMES, err


class Episode:
    def __init__(self, task: dict, enable_manual: bool = False):
        self.task = task
        rng = random.Random(task["seed"])
        self.catalog = task.get("catalog") or make_catalog(rng, heldout=task.get("heldout_entities", False))
        self.initial_state = st.clone(task["init_state"])
        self.state = st.clone(task["init_state"])
        self.allowed = set(CORE_TOOL_NAMES) | ({"search_manual"} if enable_manual else set())
        self.surfaced: set[str] = set()      # IDs the policy has legitimately seen
        self.calls: list[dict] = []          # append-only log, read by the verifier
        self.finished = False
        self.finish_payload: dict | None = None
        # one-shot recoverable faults, e.g. {"nav_start": 1}
        self._faults = dict(task.get("faults", {}))

    @property
    def budget_left(self) -> int:
        return spec.MAX_TOOL_CALLS - len(self.calls)

    def call(self, name: str, args) -> dict:
        if self.finished:
            res = err("EPISODE_FINISHED", "finish was already called")
        elif self.budget_left <= 0:
            res = err("BUDGET_EXCEEDED", f"at most {spec.MAX_TOOL_CALLS} tool calls")
        elif name not in self.allowed:
            res = err("UNKNOWN_TOOL", f"unknown tool: {name}")
        elif not isinstance(args, dict):
            res = err("INVALID_ARGUMENT", "arguments must be a JSON object")
        elif self._faults.get(name, 0) > 0:
            self._faults[name] -= 1
            res = err("SERVICE_BUSY", "temporary failure, please retry", retryable=True)
        else:
            res = self._dispatch(name, args)
        self.calls.append({"name": name, "args": args, "result": res})
        return res

    def _dispatch(self, name, args):
        fn = IMPLS[name]
        params = set(inspect.signature(fn).parameters) - {"ep"}
        extra = set(args) - params
        if extra:
            return err("INVALID_ARGUMENT", f"unexpected arguments: {sorted(extra)}")
        before = st.clone(self.state)
        try:
            res = fn(self, **args)
        except Exception as e:  # defensive: a tool bug must not crash training
            res = err("INTERNAL_ERROR", repr(e))
        if not res["ok"]:
            self.state = before               # failed calls never change state
        return res
