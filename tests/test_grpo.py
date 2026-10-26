"""GRPO sanity: positive-advantage tokens become more likely, negative ones less.

Needs the tiny smoke-test model (scripts/make_tiny_model.py).
"""
import os
import unittest

import torch

TINY = "runs/tiny-gpt2"


@unittest.skipUnless(os.path.isdir(TINY) and os.path.exists("data/train.jsonl"), "tiny model or data missing")
class GRPOSignTest(unittest.TestCase):
    def test_policy_gradient_sign(self):
        from cockpit.train.common import pad_batch, token_logprobs
        from cockpit.train.grpo import GRPOTrainer
        tr = GRPOTrainer({"model": TINY, "train_file": "data/train.jsonl", "output_dir": "runs/test_grpo",
                          "dtype": "float32", "device": "cuda" if torch.cuda.is_available() else "cpu",
                          "lr": 5e-3, "micro_batch": 2, "beta_kl": 0.0})
        good = {"role": "assistant", "content": '<tool_call>{"name": "finish"}</tool_call>'}
        bad = {"role": "assistant", "content": "我不知道"}
        base = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        samples = []
        for msg, adv in ((good, 1.0), (bad, -1.0)):
            ids, mask, _ = tr.fmt.encode(base + [msg])
            samples.append({"ids": ids, "mask": mask, "adv": adv})

        def seq_logp():
            with torch.no_grad():
                ids, att, lm = pad_batch([s["ids"] for s in samples], [s["mask"] for s in samples],
                                         tr.tok.pad_token_id)
                lp, _ = token_logprobs(tr.model, ids.to(tr.model.device), att.to(tr.model.device))
                return (lp * lm[:, 1:].to(lp.device)).sum(1).tolist()

        before = seq_logp()
        n_tok = sum(sum(s["mask"]) for s in samples)
        for _ in range(5):
            tr.precompute(samples)
            tr.opt.zero_grad()
            loss, _ = tr.loss_and_stats(samples, n_tok, len(samples))
            loss.backward()
            tr.opt.step()
        after = seq_logp()
        self.assertGreater(after[0], before[0])     # advantage +1 -> more likely
        self.assertLess(after[1], before[1])        # advantage -1 -> less likely


if __name__ == "__main__":
    unittest.main()
