"""Build a tiny random GPT-2 (2 layers) for smoke tests. No download needed.

    python scripts/make_tiny_model.py --tokenizer <dir with gpt2 tokenizer> --out runs/tiny-gpt2
"""
import argparse
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

ap = argparse.ArgumentParser()
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--out", default="runs/tiny-gpt2")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.tokenizer)
tok.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
tok.pad_token = "<|im_end|>"
cfg = GPT2Config(vocab_size=len(tok), n_positions=4096, n_embd=128, n_layer=2, n_head=4)
GPT2LMHeadModel(cfg).save_pretrained(a.out)
tok.save_pretrained(a.out)
print("saved", a.out, "vocab", len(tok))
