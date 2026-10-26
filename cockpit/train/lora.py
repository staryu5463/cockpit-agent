"""Minimal LoRA, no peft dependency.

W x  ->  W x + (alpha / r) * B(A(dropout(x)))      A: d_in -> r,  B: r -> d_out (init 0)
Only A and B train. With the adapters switched off the model is exactly the
base model, which GRPO uses as its frozen reference policy.
"""
from __future__ import annotations

import contextlib
import math

import torch
from torch import nn

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                   "c_attn", "c_proj", "c_fc")          # last three: GPT-2 names, for smoke tests


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Module, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base = base
        d_in, d_out = _dims(base)
        self.lora_A = nn.Linear(d_in, r, bias=False)
        self.lora_B = nn.Linear(r, d_out, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)        # starts as an exact no-op
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.enabled = True
        dev, dt = next(base.parameters()).device, next(base.parameters()).dtype
        self.lora_A.to(dev, dt); self.lora_B.to(dev, dt)

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + self.lora_B(self.lora_A(self.dropout(x))) * self.scale
        return y


def _dims(m):
    if isinstance(m, nn.Linear):
        return m.in_features, m.out_features
    w = m.weight                                    # GPT-2 Conv1D stores (in, out)
    return w.shape[0], w.shape[1]


def apply_lora(model: nn.Module, r=16, alpha=32, dropout=0.05, targets=DEFAULT_TARGETS):
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and hasattr(child, "weight") and child.weight.dim() == 2:
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
                n += 1
    if n == 0:
        raise ValueError("no target modules found for LoRA")
    return model


def lora_modules(model):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def lora_parameters(model):
    return [p for m in lora_modules(model) for p in (m.lora_A.weight, m.lora_B.weight)]


@contextlib.contextmanager
def lora_disabled(model):
    mods = lora_modules(model)
    for m in mods:
        m.enabled = False
    try:
        yield
    finally:
        for m in mods:
            m.enabled = True


def lora_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def load_lora_state_dict(model, sd):
    missing = model.load_state_dict(sd, strict=False)
    bad = [k for k in sd if k in missing.unexpected_keys]
    if bad:
        raise KeyError(f"LoRA keys not found in model: {bad[:3]}")


@torch.no_grad()
def merge_lora(model: nn.Module):
    """Fold every adapter into its base weight (W += scale * B A) and drop the
    wrapper, so inference runs on plain layers at no extra cost."""
    n = 0
    for module in list(model.modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                delta = (child.lora_B.weight.float() @ child.lora_A.weight.float()) * child.scale
                w = child.base.weight
                if not isinstance(child.base, nn.Linear):        # GPT-2 Conv1D stores (in, out)
                    delta = delta.T
                w.copy_((w.float() + delta).to(w.dtype))
                setattr(module, child_name, child.base)
                n += 1
    return n


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
