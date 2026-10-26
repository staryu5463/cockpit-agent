"""DPO sanity: after a few steps the chosen trajectory gains log-prob relative to the rejected one."""
import os
import unittest

import torch

TINY = "runs/tiny-gpt2"


@unittest.skipUnless(os.path.isdir(TINY), "tiny model missing")
class DPOTest(unittest.TestCase):
    def test_margin_grows(self):
        from cockpit.train.dpo import DPOTrainer, seq_logp
        tr = DPOTrainer({"model": TINY, "output_dir": "runs/test_dpo", "dtype": "float32",
                         "device": "cuda" if torch.cuda.is_available() else "cpu", "lr": 5e-3,
                         "beta": 0.5, "grad_accum": 1, "log_every": 100})
        base = [{"role": "system", "content": "s"}, {"role": "user", "content": "打开空调"}]
        good = base + [{"role": "assistant", "content": '<tool_call>{"name": "ac_power", "arguments": {"on": true}}</tool_call>'}]
        bad = base + [{"role": "assistant", "content": "好的"}]
        pairs = [{"task_id": "x", "c": tr.fmt.encode(good)[:2], "r": tr.fmt.encode(bad)[:2]}]

        def gap():
            with torch.no_grad():
                return (seq_logp(tr.model, tr.fmt, [pairs[0]["c"]]) - seq_logp(tr.model, tr.fmt, [pairs[0]["r"]])).item()

        before = gap()
        tr.cfg["epochs"] = 5
        tr.train(pairs)
        self.assertGreater(gap(), before)


if __name__ == "__main__":
    unittest.main()
