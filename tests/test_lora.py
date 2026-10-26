"""Merging LoRA into the base weights must not change the model's outputs."""
import unittest

import torch
from torch import nn

from cockpit.train.lora import LoRALinear, apply_lora, merge_lora


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 6)
        self.up_proj = nn.Linear(6, 4, bias=False)

    def forward(self, x):
        return self.up_proj(torch.relu(self.q_proj(x)))


class MergeTest(unittest.TestCase):
    def test_merge_keeps_outputs(self):
        torch.manual_seed(0)
        m = apply_lora(Toy(), r=2, alpha=4, dropout=0.0)
        for mod in m.modules():
            if isinstance(mod, LoRALinear):
                nn.init.normal_(mod.lora_B.weight)          # make the adapters non-trivial
        x = torch.randn(3, 8)
        before = m(x)
        self.assertEqual(merge_lora(m), 2)
        self.assertFalse(any(isinstance(mod, LoRALinear) for mod in m.modules()))
        torch.testing.assert_close(m(x), before, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
