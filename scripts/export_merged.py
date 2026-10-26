"""Merge a trained LoRA into the base model and save a plain Hugging Face checkpoint.

The output loads with any runtime (transformers, vLLM, llama.cpp's GGUF converter).

    python scripts/export_merged.py --lora runs/grpo_v2/lora_step40.pt --out runs/export/grpo_v2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from cockpit.train.common import load_model  # noqa: E402
from cockpit.train.lora import apply_lora, load_lora_state_dict, merge_lora  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/Qwen3-0.6B")
ap.add_argument("--lora", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--lora-r", type=int, default=16)
ap.add_argument("--lora-alpha", type=int, default=32)
ap.add_argument("--dtype", default="bfloat16")
a = ap.parse_args()

model, tok = load_model(a.model, dtype="float32", device="cpu")      # merge in fp32, cast once at the end
apply_lora(model, r=a.lora_r, alpha=a.lora_alpha, dropout=0.0)
load_lora_state_dict(model, torch.load(a.lora, map_location="cpu"))
print("merged", merge_lora(model), "layers")
model.to(getattr(torch, a.dtype)).save_pretrained(a.out)
tok.save_pretrained(a.out)
print("saved to", a.out)
