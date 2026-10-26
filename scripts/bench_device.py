"""On-device latency of the trained policy, measured together with its success rate.

Runs N real episodes (greedy) on the chosen device and reports per-turn and
per-episode latency, prefill / decode speed, peak memory and success on the same
tasks, so speed is never reported without quality.

    python scripts/bench_device.py --device cuda --dtype bfloat16 --tag gpu_bf16
    python scripts/bench_device.py --device cpu --dtype float32 --threads 8 --tag cpu_fp32_t8
    python scripts/bench_device.py --device cpu --int8 --threads 8 --tag cpu_int8_t8
    python scripts/bench_device.py --device cpu --int8 --threads 8 --prefix-cache --tag cpu_int8_t8_pc

--int8          dynamic INT8 quantization of every Linear layer (weights int8, activations fp32)
--prefix-cache  compute the KV cache of the fixed system prompt once and reuse it every turn
--gguf FILE     serve with llama.cpp instead (CPU). It keeps the KV cache of the longest common
                prefix between calls, so the system prompt and all earlier turns are never recomputed.

    python scripts/bench_device.py --gguf runs/export/gguf/grpo_v2-q4_k_m.gguf --threads 8 --tag llama_q4_t8
"""
import argparse
import copy
import gc
import io
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psutil  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from cockpit.agent.format import ChatFormat  # noqa: E402
from cockpit.agent.runner import initial_messages, run_episode  # noqa: E402
from cockpit.tasks.splits import load_split  # noqa: E402
from cockpit.train.common import HFPolicy, assistant_prefix_for, load_model  # noqa: E402
from cockpit.train.lora import apply_lora, load_lora_state_dict, merge_lora  # noqa: E402


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


class TimedPolicy(HFPolicy):
    """HFPolicy that records the latency of every turn and can reuse a prefix KV cache."""

    def __init__(self, *a, device="cpu", prefix_ids=None, **kw):
        super().__init__(*a, **kw)
        self.device, self.calls, self.rss_max = device, [], 0
        self.prefix_ids, self.prefix_kv = prefix_ids, None
        if prefix_ids:
            with torch.no_grad():
                x = torch.tensor([prefix_ids], device=self.model.device)
                self.prefix_kv = self.model(x, use_cache=True).past_key_values

    @torch.no_grad()
    def act(self, messages):
        ids, _, _ = self.fmt.encode(messages, add_generation_prompt=True)
        budget = min(self.max_new_tokens, self.model.config.max_position_embeddings - len(ids))
        x = torch.tensor([ids], device=self.model.device)
        kw = self._gen_kwargs(budget)
        if self.prefix_kv is not None:
            assert ids[:len(self.prefix_ids)] == self.prefix_ids, "prompt does not start with the cached prefix"
            kw["past_key_values"] = copy.deepcopy(self.prefix_kv)
        sync(self.device)
        t0 = time.perf_counter()
        out = self.model.generate(x, attention_mask=torch.ones_like(x), **kw)
        sync(self.device)
        gen = out[0, len(ids):].tolist()
        self.calls.append({"ms": 1000 * (time.perf_counter() - t0), "prompt": len(ids), "new": len(gen)})
        self.rss_max = max(self.rss_max, psutil.Process().memory_info().rss)
        return self._cut(gen)


class LlamaCppPolicy:
    """The HFPolicy interface served by llama.cpp; token ids come from the same tokenizer."""

    def __init__(self, gguf, fmt, threads, max_new_tokens=128, n_ctx=4096):
        from llama_cpp import Llama
        self.llm = Llama(model_path=gguf, n_ctx=n_ctx, n_threads=threads, n_threads_batch=threads,
                         n_batch=512, verbose=False)
        self.fmt, self.max_new_tokens = fmt, max_new_tokens
        self.calls, self.rss_max = [], 0

    def reset(self, task):
        pass

    def act(self, messages):
        ids, _, _ = self.fmt.encode(messages, add_generation_prompt=True)
        budget = min(self.max_new_tokens, self.llm.n_ctx() - len(ids))
        gen, ttft = [], None
        t0 = time.perf_counter()
        for tok in self.llm.generate(ids, top_k=1, top_p=1.0, min_p=0.0, temp=0.0, repeat_penalty=1.0):
            if ttft is None:
                ttft = time.perf_counter() - t0
            gen.append(tok)
            if tok == self.fmt.im_end_id or len(gen) >= budget:
                break
        total = time.perf_counter() - t0
        self.calls.append({"ms": 1000 * total, "prompt": len(ids), "new": len(gen), "ttft_ms": 1000 * ttft,
                           "decode_ms_per_tok": 1000 * (total - ttft) / max(len(gen) - 1, 1)})
        self.rss_max = max(self.rss_max, psutil.Process().memory_info().rss)
        return HFPolicy._cut(self, gen)


@torch.no_grad()
def speed_probe(model, fmt, tasks, device, n=5, decode_tokens=32):
    """Prefill tokens/s on the first-turn prompt and decode ms/token for a fixed length."""
    pre, dec = [], []
    for t in tasks[:n]:
        ids, _, _ = fmt.encode(initial_messages(t), add_generation_prompt=True)
        x = torch.tensor([ids], device=model.device)
        sync(device); t0 = time.perf_counter()
        model(x, use_cache=True)
        sync(device); t_pre = time.perf_counter() - t0
        t0 = time.perf_counter()
        model.generate(x, attention_mask=torch.ones_like(x), do_sample=False, temperature=None, top_p=None,
                       top_k=None, min_new_tokens=decode_tokens, max_new_tokens=decode_tokens,
                       pad_token_id=fmt.im_end_id)
        sync(device); t_gen = time.perf_counter() - t0
        pre.append(len(ids) / t_pre)
        dec.append(1000 * max(t_gen - t_pre, 1e-6) / decode_tokens)
    return statistics.median(pre), statistics.median(dec)


def weights_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-0.6B")
    ap.add_argument("--lora", default="runs/grpo_v2/lora_step40.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--prefix-cache", action="store_true")
    ap.add_argument("--split", default="test_expr")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--hybrid", action="store_true", help="confident simple commands take the rule fast path")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    if a.gguf:
        return main_llamacpp(a)

    torch.set_num_threads(a.threads)
    model, tok = load_model(a.model, dtype=a.dtype, device=a.device)
    if a.lora:
        apply_lora(model, r=16, alpha=32, dropout=0.0)
        load_lora_state_dict(model, torch.load(a.lora, map_location="cpu"))
        merge_lora(model)
    model.eval()
    if a.int8:
        assert a.device == "cpu", "dynamic INT8 runs on CPU"
        model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8, inplace=True)
        gc.collect()
    fmt = ChatFormat(tok, assistant_prefix_for(a.model))
    tasks = load_split(f"data/{a.split}.jsonl")[: a.n]

    prefix = None
    if a.prefix_cache:
        prefix = fmt.encode(initial_messages(tasks[0])[:1])[0]        # the system block, same for every task
    pol = TimedPolicy(model, fmt, greedy=True, max_new_tokens=a.max_new_tokens, device=a.device,
                      prefix_ids=prefix)

    prefill_tps, decode_ms = speed_probe(model, fmt, tasks, a.device)
    run_episode(tasks[0], pol)                                       # warm-up
    pol.calls.clear()
    if a.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    eps, rows = [], []
    for t in tasks:
        n0 = len(pol.calls)
        r = run_episode(t, pol)
        ms = sum(c["ms"] for c in pol.calls[n0:])
        eps.append(ms)
        rows.append({"task_id": t["id"], "tier": t["tier"], "success": bool(r["verify"]["success"]),
                     "turns": len(pol.calls) - n0, "ms": round(ms, 1)})
    turns = [c["ms"] for c in pol.calls]
    # GPU: peak allocated by torch; CPU: peak resident memory sampled after every turn
    peak = (torch.cuda.max_memory_allocated() if a.device == "cuda" else pol.rss_max) / 2**20
    res = {"tag": a.tag, "device": a.device, "dtype": "int8-dynamic" if a.int8 else a.dtype,
           "threads": a.threads if a.device == "cpu" else None, "prefix_cache": a.prefix_cache,
           "split": a.split, "n": len(tasks),
           "success": statistics.fmean(r["success"] for r in rows),
           "turn_p50_ms": pct(turns, .5), "turn_p95_ms": pct(turns, .95),
           "episode_p50_ms": pct(eps, .5), "episode_p95_ms": pct(eps, .95),
           "turns_per_episode": len(turns) / len(tasks),
           "new_tokens_per_turn": statistics.fmean(c["new"] for c in pol.calls),
           "prompt_tokens_per_turn": statistics.fmean(c["prompt"] for c in pol.calls),
           "prefill_tok_per_s": prefill_tps, "decode_ms_per_tok": decode_ms,
           "peak_mem_mb": peak, "weights_mb": weights_mb(model),
           "cpu": psutil.cpu_count(logical=True), "torch": torch.__version__}
    save(a, res, rows)


def run_tasks(pol, tasks, router=None):
    runner = router or pol
    run_episode(tasks[0], runner)                                    # warm-up
    pol.calls.clear()
    if router:
        router.routed, router.flags = {"fast": 0, "agent": 0}, []
    eps, rows = [], []
    for t in tasks:
        n0 = len(pol.calls)
        r = run_episode(t, runner)
        # with a router, time the whole episode: fast-path episodes make no model calls
        ms = 1000 * r["seconds"] if router else sum(c["ms"] for c in pol.calls[n0:])
        eps.append(ms)
        rows.append({"task_id": t["id"], "tier": t["tier"], "success": bool(r["verify"]["success"]),
                     "turns": len(pol.calls) - n0, "ms": round(ms, 1)})
    return eps, rows


def main_llamacpp(a):
    from transformers import AutoTokenizer
    import llama_cpp
    fmt = ChatFormat(AutoTokenizer.from_pretrained(a.model), assistant_prefix_for(a.model))
    tasks = load_split(f"data/{a.split}.jsonl")[: a.n]
    pol = LlamaCppPolicy(a.gguf, fmt, a.threads, max_new_tokens=a.max_new_tokens)
    router = None
    if a.hybrid:
        from cockpit.serve.router import HybridPolicy
        router = HybridPolicy(pol)
    eps, rows = run_tasks(pol, tasks, router)
    turns = [c["ms"] for c in pol.calls]
    res = {"tag": a.tag, "device": "cpu", "backend": "llama.cpp " + llama_cpp.__version__,
           "dtype": Path(a.gguf).stem.split("-")[-1], "threads": a.threads, "prefix_cache": "all turns",
           "split": a.split, "n": len(tasks),
           "success": statistics.fmean(r["success"] for r in rows),
           "turn_p50_ms": pct(turns, .5), "turn_p95_ms": pct(turns, .95),
           "episode_p50_ms": pct(eps, .5), "episode_p95_ms": pct(eps, .95),
           "turns_per_episode": len(turns) / len(tasks),
           "new_tokens_per_turn": statistics.fmean(c["new"] for c in pol.calls),
           "prompt_tokens_per_turn": statistics.fmean(c["prompt"] for c in pol.calls),
           "ttft_p50_ms": pct([c["ttft_ms"] for c in pol.calls], .5),
           "decode_ms_per_tok": statistics.median(c["decode_ms_per_tok"] for c in pol.calls if c["new"] > 1),
           "peak_mem_mb": pol.rss_max / 2**20, "weights_mb": Path(a.gguf).stat().st_size / 2**20,
           "cpu": psutil.cpu_count(logical=True)}
    if router:
        res["fast_path_share"] = router.routed["fast"] / len(tasks)
        res["fast_path_success"] = (statistics.fmean(r["success"] for r, f in zip(rows, router.flags) if f)
                                    if any(router.flags) else None)
    save(a, res, rows)


def save(a, res, rows):
    res = {k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}
    print(json.dumps(res, indent=2))
    out = Path("runs/bench")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "device.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")
    with open(out / f"episodes_{a.tag}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
