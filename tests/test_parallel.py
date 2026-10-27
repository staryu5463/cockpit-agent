"""Compact call style, multi-call turns and the parallel expert schedule."""
import json
import os
import unittest
from unittest import mock

from cockpit.agent import format as F
from cockpit.agent.format import format_tool_call, format_turn, parse_tool_call, parse_tool_calls
from cockpit.agent.runner import execute_turn, run_episode
from cockpit.baselines.planner import ExpertPolicy
from cockpit.env.episode import Episode
from cockpit.tasks.splits import load_split


class CompactStyleTest(unittest.TestCase):
    def test_round_trip(self):
        for args in ({}, {"value": 8}, {"zone": "driver", "value": 22},
                     {"query": 'a,b=(c)"d', "on": False, "fields": ["ac.temp.driver", "media.volume"]},
                     {"status": "done", "message": "已开始导航到西贝莜面村(闵行餐厅)，温度已调整。"}):
            self.assertEqual(parse_tool_call(format_tool_call("t", args, "fn")), ("t", args, None))

    def test_shorter_than_json(self):
        args = {"zone": "driver", "value": 22}
        self.assertLess(len(format_tool_call("ac_set_temperature", args, "fn")),
                        len(format_tool_call("ac_set_temperature", args, "json")))

    def test_bad_calls_are_errors(self):
        for text in ("<tool_call>bad(</tool_call>", "<tool_call>f(x)</tool_call>", "<tool_call>f(x=1 y=2)</tool_call>"):
            self.assertIsNotNone(parse_tool_call(text)[2], text)
        self.assertEqual(parse_tool_call("ac_power(on=True)")[2], "no <tool_call> block")

    def test_several_calls_in_one_turn(self):
        text = format_turn([("search_poi", {"query": "加油站"}), ("get_vehicle_status", {"fields": ["media.volume"]})], "fn")
        self.assertEqual([c[0] for c in parse_tool_calls(text)], ["search_poi", "get_vehicle_status"])


@unittest.skipUnless(os.path.exists("data/test_combo.jsonl"), "data not generated")
class ParallelTurnTest(unittest.TestCase):
    def setUp(self):
        self.task = load_split("data/test_combo.jsonl")[0]

    def test_finish_must_be_alone(self):
        ep = Episode(self.task)
        text = format_turn([("ac_power", {"on": True}), ("finish", {"status": "done", "message": "好"})])
        content, bad = execute_turn(ep, text, parallel=True)
        self.assertTrue(bad)
        self.assertEqual(json.loads(content)["code"], "FORMAT_ERROR")
        self.assertEqual(ep.calls, [])                        # nothing was executed

    def test_single_mode_keeps_first_call_only(self):
        ep = Episode(self.task)
        text = format_turn([("get_vehicle_status", {"fields": ["media.volume"]}), ("ac_power", {"on": True})])
        execute_turn(ep, text, parallel=False)
        self.assertEqual([c["name"] for c in ep.calls], ["get_vehicle_status"])

    @mock.patch.object(F, "PARALLEL", True)              # the runner executes every call of a turn
    def test_parallel_expert_matches_sequential(self):
        for split in ("test_id", "test_combo"):
            for t in load_split(f"data/{split}.jsonl")[:60]:
                seq = run_episode(t, ExpertPolicy(parallel=False))
                par = run_episode(t, ExpertPolicy(parallel=True))
                self.assertEqual(par["verify"]["success"], seq["verify"]["success"], t["id"])
                self.assertEqual(par["episode"].state, seq["episode"].state, t["id"])
                self.assertLessEqual(par["verify"]["turns"], seq["verify"]["turns"])


if __name__ == "__main__":
    unittest.main()
