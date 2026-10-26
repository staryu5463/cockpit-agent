"""Hybrid router: cheap rule NLU for simple commands, the agent for the rest.

The question it answers: what share of real requests can stay on the
millisecond-level fast path, and what does routing cost in accuracy?

A request goes to the fast path only if the rule parser is confident:
exactly one op, an absolute setting or power switch, and the parsed text
covers the whole sentence (no leftover words it did not understand).
"""
from __future__ import annotations

import re

from cockpit.baselines.rule_nlu import RulePolicy, parse

FAST_OPS = {"AC_ON", "AC_OFF", "TEMP_ABS", "VOL_ABS"}
FILLER = re.compile(r"[把将给帮我请麻烦一下吧呢啊呀，。！ ]")


def fast_path_ok(text: str) -> bool:
    ops = parse(text)
    if len(ops) != 1 or ops[0]["op"] not in FAST_OPS:
        return False
    return len(FILLER.sub("", text)) <= 14            # short, single-intent utterance


class HybridPolicy:
    def __init__(self, agent_policy):
        self.agent = agent_policy
        self.rule = RulePolicy()
        self.routed = {"fast": 0, "agent": 0}
        self.flags = []                                # per episode: True if it took the fast path

    def reset(self, task):
        self.use_fast = fast_path_ok(task["text"])
        self.routed["fast" if self.use_fast else "agent"] += 1
        self.flags.append(self.use_fast)
        (self.rule if self.use_fast else self.agent).reset(task)

    def act(self, messages):
        return (self.rule if self.use_fast else self.agent).act(messages)
