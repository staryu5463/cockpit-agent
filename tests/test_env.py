"""Environment, verifier, generator and baseline tests (stdlib unittest).

Run:  python -m unittest discover -s tests -v
"""
import copy
import random
import unittest

from cockpit import spec
from cockpit.agent.runner import run_episode
from cockpit.baselines.planner import ExpertPolicy
from cockpit.baselines.rule_nlu import RulePolicy, parse
from cockpit.env.episode import Episode
from cockpit.tasks.generator import make_task
from cockpit.tasks.splits import build_split
from cockpit.verifier import verify


def _task(tier="L1", family="A", seed=0):
    return make_task(random.Random(seed), tier, family, task_id=f"t{seed}")


class EnvTest(unittest.TestCase):
    def test_failed_call_does_not_change_state(self):
        t = _task()
        ep = Episode(t)
        before = dict(ep.state)
        r = ep.call("ac_set_temperature", {"zone": "all", "value": 99})
        self.assertFalse(r["ok"])
        self.assertEqual(r["code"], "OUT_OF_RANGE")
        self.assertEqual(ep.state, before)

    def test_unknown_id_rejected(self):
        """IDs must come from a search in this episode: no guessing."""
        t = _task("L3", seed=3)
        ep = Episode(t)
        pid = t["catalog"]["pois"][0]["id"]
        self.assertEqual(ep.call("nav_start", {"poi_id": pid})["code"], "UNKNOWN_ID")

    def test_unexpected_argument(self):
        ep = Episode(_task())
        self.assertEqual(ep.call("volume_set", {"value": 5, "x": 1})["code"], "INVALID_ARGUMENT")

    def test_budget(self):
        ep = Episode(_task())
        for _ in range(spec.MAX_TOOL_CALLS):
            ep.call("get_vehicle_status", {})
        self.assertEqual(ep.call("get_vehicle_status", {})["code"], "BUDGET_EXCEEDED")

    def test_transient_fault_is_retryable_once(self):
        t = _task("L3", seed=5)
        t["faults"] = {"search_poi": 1}
        ep = Episode(t)
        r1 = ep.call("search_poi", {"query": "充电站"})
        r2 = ep.call("search_poi", {"query": "充电站"})
        self.assertTrue(r1["retryable"] and not r1["ok"])
        self.assertTrue(r2["ok"])


class VerifierTest(unittest.TestCase):
    def test_side_effect_fails(self):
        t = _task("L1", seed=1)
        ep = Episode(t)
        ExpertPolicy  # noqa
        res = run_episode(t, ExpertPolicy())
        self.assertTrue(res["verify"]["success"], res["verify"])
        ep = res["episode"]
        ep.state["nav.destination"] = "p0000"          # tamper: extra change
        self.assertFalse(verify(ep)["success"])

    def test_claiming_done_is_not_enough(self):
        t = _task("L1", seed=2)
        ep = Episode(t)
        ep.call("finish", {"status": "done", "message": "好了"})
        self.assertFalse(verify(ep)["success"])


class CounterexampleTest(unittest.TestCase):
    """Two checks from the design doc that the benchmark must pass."""

    def test_relative_temperature_depends_on_state(self):
        rng = random.Random(7)
        for _ in range(200):
            t = make_task(rng, "L4", "A", task_id="x")
            if t["semantic"][0]["op"] == "TEMP_REL" and t["semantic"][0]["zone"] == "driver":
                break
        t2 = copy.deepcopy(t)
        t2["init_state"]["ac.temp.driver"] += 1 if t2["init_state"]["ac.temp.driver"] < 28 else -1
        from cockpit.tasks.generator import _apply
        g1 = _apply(t["semantic"][0], t["init_state"], t["catalog"])
        g2 = _apply(t2["semantic"][0], t2["init_state"], t2["catalog"])
        self.assertNotEqual(g1, g2)                    # same sentence, different target
        t2["goal"]["fields"] = g2
        self.assertTrue(run_episode(t2, ExpertPolicy())["verify"]["success"])

    def test_navigation_follows_search_results(self):
        rng = random.Random(11)
        for _ in range(200):
            t = make_task(rng, "L3", "A", task_id="x")
            if t["semantic"][0]["op"] == "NAV_NEAREST":
                break
        # shuffle the catalog order: the expert must still succeed
        t2 = copy.deepcopy(t)
        random.Random(1).shuffle(t2["catalog"]["pois"])
        self.assertTrue(run_episode(t2, ExpertPolicy())["verify"]["success"])


class GeneratorTest(unittest.TestCase):
    def test_expert_solves_every_split(self):
        for name in ["train", "test_expr", "test_entity", "test_combo"]:
            tasks = build_split(name, 120, seed=0)
            fails = [(t["id"], t["text"], r["verify"]["reasons"])
                     for t in tasks for r in [run_episode(t, ExpertPolicy())]
                     if not r["verify"]["success"]]
            self.assertEqual(fails, [], f"{name}: expert failed {fails[:3]}")

    def test_combo_split_is_heldout(self):
        from cockpit.tasks.generator import HELDOUT_COMBOS
        train = build_split("train", 300, seed=0)
        for t in train:
            names = frozenset(o["op"] for o in t["semantic"])
            self.assertFalse(any(c <= names for c in HELDOUT_COMBOS), t["text"])

    def test_rule_parser_examples(self):
        self.assertEqual(parse("把主驾空调温度调到22度")[0],
                         {"op": "TEMP_ABS", "zone": "driver", "value": 22})
        self.assertEqual(parse("声音小一点")[0], {"op": "VOL_REL", "delta": -spec.DEFAULT_VOLUME_STEP})
        self.assertEqual(parse("导航去最近的充电站")[0], {"op": "NAV_NEAREST", "category": "charging"})
        self.assertEqual(parse("今天好累啊")[0]["op"], "CHAT")


class RuleBaselineTest(unittest.TestCase):
    def test_rule_baseline_runs(self):
        tasks = build_split("dev", 60, seed=0)
        ok = sum(run_episode(t, RulePolicy())["verify"]["success"] for t in tasks)
        self.assertGreater(ok, 20)                    # sanity, not a target


if __name__ == "__main__":
    unittest.main()
