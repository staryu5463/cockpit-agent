"""Model loading, the HF policy used for rollouts, and token log-probs."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from cockpit.agent.format import IM_END, IM_START, ChatFormat

QWEN3_NO_THINK = "<think>\n\n</think>\n\n"


def load_model(path, dtype="bfloat16", device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    if tok.convert_tokens_to_ids(IM_END) in (None, tok.unk_token_id):
        # non-ChatML tokenizer (GPT-2 smoke tests): add the two control tokens
        tok.add_special_tokens({"additional_special_tokens": [IM_START, IM_END]})
    if tok.pad_token is None:
        tok.pad_token = IM_END
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=getattr(torch, dtype))
    if model.get_input_embeddings().weight.shape[0] < len(tok):
        model.resize_token_embeddings(len(tok))
    return model.to(device), tok


def assistant_prefix_for(path_or_name: str) -> str:
    return QWEN3_NO_THINK if "qwen3" in str(path_or_name).lower() else ""


class HFPolicy:
    """Samples one assistant turn with `model.generate`, keeping the token ids."""

    def __init__(self, model, fmt: ChatFormat, temperature=1.0, top_p=1.0, max_new_tokens=256,
                 greedy=False):
        self.model, self.fmt = model, fmt
        self.temperature, self.top_p = temperature, top_p
        self.max_new_tokens, self.greedy = max_new_tokens, greedy

    def reset(self, task):
        pass

    def _gen_kwargs(self, budget):
        kw = dict(max_new_tokens=budget, eos_token_id=self.fmt.im_end_id, pad_token_id=self.fmt.im_end_id)
        if self.greedy:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        else:
            # sample from the full distribution (top_k=0 overrides the model's generation_config),
            # so the sampling policy is the same distribution the RL loss computes log-probs of
            kw.update(do_sample=True, temperature=self.temperature, top_p=self.top_p, top_k=0)
        return kw

    def _cut(self, gen):
        if self.fmt.im_end_id in gen:
            gen = gen[:gen.index(self.fmt.im_end_id) + 1]
        text = self.fmt.tok.decode([t for t in gen if t != self.fmt.im_end_id], skip_special_tokens=False)
        return {"text": text, "token_ids": gen}

    @torch.no_grad()
    def act_batch(self, messages_list):
        """One generate call for many conversations (left padding)."""
        prompts = [self.fmt.encode(m, add_generation_prompt=True)[0] for m in messages_list]
        L = max(map(len, prompts))
        limit = getattr(self.model.config, "max_position_embeddings", None) or 10**9
        budget = min(self.max_new_tokens, limit - L)
        if budget <= 0:
            return [{"text": "", "token_ids": [self.fmt.im_end_id]} for _ in prompts]
        pad = self.fmt.im_end_id
        ids = torch.full((len(prompts), L), pad, dtype=torch.long)
        att = torch.zeros((len(prompts), L), dtype=torch.long)
        for i, p in enumerate(prompts):
            ids[i, L - len(p):] = torch.tensor(p)
            att[i, L - len(p):] = 1
        out = self.model.generate(ids.to(self.model.device), attention_mask=att.to(self.model.device),
                                  **self._gen_kwargs(budget))
        return [self._cut(row[L:].tolist()) for row in out]

    @torch.no_grad()
    def act(self, messages):
        ids, _, _ = self.fmt.encode(messages, add_generation_prompt=True)
        limit = getattr(self.model.config, "max_position_embeddings", None) or 10**9
        budget = min(self.max_new_tokens, limit - len(ids))
        if budget <= 0:                     # context full: end the turn instead of crashing
            return {"text": "", "token_ids": [self.fmt.im_end_id]}
        x = torch.tensor([ids], device=self.model.device)
        out = self.model.generate(x, attention_mask=torch.ones_like(x), **self._gen_kwargs(budget))
        return self._cut(out[0, len(ids):].tolist())


def _chunk_logp(lg, tgt):
    return F.log_softmax(lg.float(), dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)


def token_logprobs(model, input_ids, attention_mask, chunk=512):
    """log p(x_t | x_<t) for t >= 1 and the per-position entropy, both (B, T-1).

    Logits stay in the model dtype; the float32 log-softmax runs one chunk of
    positions at a time and, when gradients are on, is recomputed in backward
    (checkpoint) instead of being stored. With a ~150k vocabulary a full float32
    log-softmax over a 3k-token sequence would otherwise need ~2 GB.
    """
    from torch.utils.checkpoint import checkpoint
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
    tgt = input_ids[:, 1:]
    lps, ents = [], []
    for s in range(0, logits.shape[1], chunk):
        lg, tg = logits[:, s:s + chunk], tgt[:, s:s + chunk]
        if torch.is_grad_enabled():
            lps.append(checkpoint(_chunk_logp, lg, tg, use_reentrant=False))
        else:
            lps.append(_chunk_logp(lg, tg))
        with torch.no_grad():
            lsm = F.log_softmax(lg.float(), dim=-1)
            ents.append(-(lsm.exp() * lsm).sum(-1))
    return torch.cat(lps, 1), torch.cat(ents, 1)


def pad_batch(seqs, masks, pad_id):
    T = max(map(len, seqs))
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), T), dtype=torch.long)
    lm = torch.zeros((len(seqs), T), dtype=torch.float)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
        lm[i, :len(m)] = torch.tensor(m, dtype=torch.float)
    return ids, att, lm
