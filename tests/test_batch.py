"""Batched rollouts must give the same greedy result as one-at-a-time rollouts."""
import os
import unittest

import torch

TINY = "runs/tiny-gpt2"


@unittest.skipUnless(os.path.isdir(TINY) and os.path.exists("data/dev.jsonl"), "tiny model or data missing")
class BatchRunnerTest(unittest.TestCase):
    def test_batch_matches_single(self):
        from cockpit.agent.batch_runner import run_episodes_batch
        from cockpit.agent.format import ChatFormat
        from cockpit.agent.runner import run_episode
        from cockpit.tasks.splits import load_split
        from cockpit.train.common import HFPolicy, load_model
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model, tok = load_model(TINY, dtype="float32", device=dev)
        pol = HFPolicy(model.eval(), ChatFormat(tok), greedy=True, max_new_tokens=12)
        tasks = load_split("data/dev.jsonl")[:3]
        single = [run_episode(t, pol, max_turns=2) for t in tasks]
        batch = run_episodes_batch(tasks, pol, max_turns=2)
        for s, b in zip(single, batch):
            sa = [m["token_ids"] for m in s["messages"] if m["role"] == "assistant"]
            ba = [m["token_ids"] for m in b["messages"] if m["role"] == "assistant"]
            self.assertEqual(sa, ba)
            self.assertEqual(s["verify"]["success"], b["verify"]["success"])


if __name__ == "__main__":
    unittest.main()
