# CockpitAgent

**A small on-device model that completes multi-step in-car voice commands by calling vehicle tools, trained with verifiable-reward RL.**
端侧智能座舱语音智能体：小模型通过多轮工具调用完成复合车控指令。

In-car assistants usually parse speech with an NLU model plus hand-written rules. Two things break in practice:
the rules fail as soon as a driver phrases a request differently, and the assistant often says "done" when the car
state is actually wrong. This project builds a benchmark that measures both problems, then trains a 0.6B model
(Qwen3-0.6B + LoRA) with SFT and multi-turn GRPO / DAPO to fix them, on a single 8 GB laptop GPU.

## What is in the repo

| Part | What it does | Code |
|---|---|---|
| **CockpitBench** | Simulated cockpit with 10 tools in MCP `inputSchema` format (climate, volume, POI search, navigation, media, vehicle status, owner's-manual search, `finish`). A private verifier scores the *final vehicle state*, side effects and the finish status, never the model's own claim. Held-out test splits for unseen phrasings, unseen entities and unseen combinations. | `cockpit/env` `cockpit/verifier.py` `cockpit/tasks` `cockpit/eval` |
| **Baselines** | A fair scripted expert (gets information only through tools) and a regex rule-NLU + planner baseline. | `cockpit/baselines` |
| **Training** | Own LoRA, SFT on expert trajectories, multi-turn GRPO with DAPO switches (clip-higher, dynamic sampling, token-level loss, truncation filter), DPO, hard-task mining, batched multi-turn rollouts, failure attribution. | `cockpit/train` `cockpit/agent` `scripts/` |
| **Serving** | LoRA merge and GGUF export, llama.cpp CPU inference (Q8_0 / Q4_K_M) with cross-turn KV reuse, rule fast path for simple commands, vLLM client, MCP stdio server, latency benchmark that always reports success on the same tasks. | `cockpit/serve` `scripts/export_merged.py` `scripts/bench_device.py` |

### Design choices worth knowing

- **Token-level rendering.** Every conversation segment is tokenized separately and concatenated, and the token ids the
  model actually sampled are stored and reused for training. Re-rendering the whole conversation with a chat template at
  every turn can change the tokens of earlier turns, so the loss mask stops matching the sampled tokens and tool-calling
  turns silently drop out of the loss (the failure mode reported in huggingface/trl#7503).
  `tests/test_format.py::test_generated_ids_are_kept_verbatim` checks this.
- **Loss only on generated tokens.** Tool results are context, never targets; otherwise the model learns to predict
  (i.e. invent) tool outputs.
- **False completion as a first-class metric.** `false_done` = the agent called `finish(status="done")` but the
  verifier says the task failed: it told the user the job was done, and it was not.
- **Hard-task mining.** After SFT, 59% of training prompts were solved by all 8 samples, so their GRPO advantages are
  zero and give no gradient. `scripts/mine_hard.py` keeps only prompts the SFT policy sometimes fails (153 of 400);
  the same rollouts yield the DPO preference pairs.

## Results

Qwen3-0.6B, LoRA r=16, RTX 4070 Laptop (8 GB). 300 tasks per test split, the same tasks for every method,
greedy decoding. Brackets: 95% Wilson interval.

| Method | In-distribution | Unseen phrasing | Unseen entity | Unseen combination | False completion (phrasing / combination) |
|---|---:|---:|---:|---:|---:|
| Rule NLU + planner | 92.3 [88.8, 94.8] | 24.7 [20.1, 29.8] | 88.7 [84.6, 91.8] | 84.7 [80.2, 88.3] | 45.3% / 12.3% |
| Zero-shot Qwen3-0.6B | 4.7 | 2.3 | — | — | 3.0% / — |
| SFT (LoRA) | 94.0 [90.7, 96.2] | 80.7 [75.8, 84.7] | 89.0 [85.0, 92.1] | 80.3 [75.5, 84.4] | 17.0% / 16.3% |
| SFT → DPO | 94.7 [91.5, 96.7] | 80.0 [75.1, 84.1] | 90.7 [86.8, 93.5] | 81.0 [76.2, 85.0] | 18.3% / 16.7% |
| **SFT → GRPO / DAPO** | **95.0** [91.9, 96.9] | **83.3** [78.7, 87.1] | **93.0** [89.5, 95.4] | **86.3** [82.0, 89.8] | **15.7% / 13.0%** |
| SFT with 30% parallel-call data → GRPO / DAPO ¹ | 96.0 [93.1, 97.7] | **88.3** [84.2, 91.5] | 91.3 [87.6, 94.0] | 85.3 [80.9, 88.9] | **11.3% / 14.3%** |

¹ From the efficiency experiments below. Against the row above: +5.0 pt on unseen phrasing (p = 0.014),
no significant difference elsewhere. One seed, and the cause (more varied SFT data) is not isolated yet.

**Paired tests** (McNemar exact test on the tasks where two methods disagree; full table in `results/stats.md`):

| Comparison | Unseen phrasing | Unseen entity | Unseen combination |
|---|---:|---:|---:|
| GRPO vs SFT | +2.7 pt, p = 0.077 | **+4.0 pt, p = 0.023** | **+6.0 pt, p = 0.004** |
| GRPO vs DPO (same mined data) | **+3.3 pt, p = 0.031** | +2.3 pt, p = 0.17 | **+5.3 pt, p = 0.009** |
| DPO vs SFT | −0.7 pt, p = 0.63 | +1.7 pt, p = 0.13 | +0.7 pt, p = 0.77 |
| GRPO vs rules | **+58.7 pt, p < 1e-50** | +4.3 pt, p = 0.066 | +1.7 pt, p = 0.67 |

What the numbers support, and what they do not:

- Learned policies generalize to new phrasings where rules collapse (24.7% → 80–83%), and they cut false completions
  from 45.3% to 15.7%.
- GRPO on mined hard tasks beats SFT on unseen entities and combinations (31% fewer combination failures, 59 → 41),
  mostly by fixing wrong-location picks and skipped status reads.
- DPO on the same mined data is indistinguishable from SFT here. Its 164 fixed pairs give far less signal than ~1,300
  on-policy rollouts, and two multi-turn trajectories usually differ in one step, so the sequence-level margin is diluted.
- On in-distribution and unseen-combination tasks GRPO is **not** significantly better than the rule system.
- All model rows are a single training seed; the intervals above cover test-set sampling only.

### Efficiency: parallel tool calls and an efficiency reward

**Where the time goes.** On CPU a command costs about `turns × tokens per turn × time per token`. The trained policy
already uses almost the fewest tool calls (0.15 more than the expert per command on unseen phrasing), so there are no
wasted steps to remove. The two real levers are tokens per call and the number of turns.

**What was built** (`COCKPIT_PARALLEL=1`, `COCKPIT_CALL_STYLE=json|fn`):
- Parallel turns: up to 4 independent calls in one turn, results returned together in order. `finish` must be
  alone, so the agent cannot report "done" before it has seen the results of its own actions.
- A parallel expert schedule: calls are grouped by dependency level (a search before the navigation that uses its id,
  a status read before a relative change) and replayed to check the final state matches the sequential plan. The
  fewest possible turns drop from 6.14 to 3.18 on unseen combinations, with identical success.
- SFT where 30% of tasks use the parallel schedule, then GRPO with a turn penalty
  `0.1 × max(0, turns − expert turns)` applied **only to successful episodes**.

Same tasks, paired McNemar tests, one seed (full tables: `results/efficiency.md`):

| Split | Success: no penalty → turn penalty | Turns: no penalty → turn penalty |
|---|---:|---:|
| In-distribution | 96.0% → 94.7% (p = 0.45) | 3.33 → 2.63 (**−21%**) |
| Unseen phrasing | 88.3% → 86.3% (p = 0.24) | 3.26 → 2.67 (**−18%**) |
| Unseen entity | 91.3% → 90.0% (p = 0.54) | 4.39 → 3.19 (**−27%**) |
| Unseen combination | 85.3% → **77.7% (p < 0.001)** | 6.17 → 3.26 (**−47%**) |

**Fewer turns did not make the CPU much faster.** llama.cpp Q8_0, 8 threads, first 100 tasks of each split:

| Split | Turns | Tokens per turn | Command P50 | Success |
|---|---:|---:|---:|---:|
| Unseen phrasing, no penalty → turn penalty | 3.31 → 2.67 | 25.9 → 32.3 | 1.57 s → 1.61 s | 83% → 84% |
| Unseen combination, no penalty → turn penalty | 6.12 → 3.33 | 25.7 → 46.4 | 3.62 s → 3.39 s (−6%) | 88% → 75% |

Parallel turns carry the same calls, so the generated tokens barely change, and on a CPU decoding those tokens is
almost all of the time. Merging turns only saves the per-turn overhead (reading the new tool results, starting a
generation), which llama.cpp already keeps small by reusing the KV cache. In this benchmark the tools are simulated
and answer instantly; parallel calls pay off when tools have real latency (a cloud POI search, a vehicle bus
round-trip), because their waits overlap. That needs a measurement with realistic tool latency before any speed
claim.

What did not work, and why:
- **A compact call style** (`ac_set_temperature(zone="driver",value=28)` instead of JSON) cut generated tokens by 40%
  but lost 13.7 points on unseen phrasings (p < 1e-7), mostly misunderstood requests. Qwen3 was pretrained on JSON
  tool calls; moving away from that format costs more than the tokens it saves.
- **A turn penalty on every episode was exploited.** The policy skipped retries after recoverable errors and rushed
  its choices, trading success for fewer turns (unseen combinations 85.3% → 71.7%). Gating the penalty on success
  removed most of that loss.
- **The remaining loss on combinations** comes from wrong choices and from missed failures inside batched results:
  when one of several results in a turn is an error, the model sometimes does not retry it. Flagging failed calls in
  batched results is the next fix.

### On-device latency

`scripts/bench_device.py` runs full episodes on the target device and reports latency together with success on the
same tasks.

Same trained model (SFT → GRPO, LoRA merged), the same 100 unseen-phrasing tasks, greedy decoding.
Laptop CPU: Intel i7-12800HX; GPU: RTX 4070 Laptop. Latency covers model calls only (tools are simulated).

| Runtime | Precision | Threads | Command P50 / P95 (s) | Turn P50 (ms) | Decode (ms / token) | Model file | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| PyTorch eager, GPU | BF16 | — | 2.68 / 5.96 | 866 | 32.0 | 1.11 GB | 79% |
| PyTorch eager, CPU | FP32 | 8 | 12.56 / 32.96 ¹ | 4217 | 73.0 | 2.22 GB | 83% ¹ |
| PyTorch eager, CPU + system-prompt KV cache | FP32 | 8 | 7.98 / 21.22 | 2609 | 77.1 | 2.22 GB | 79% |
| **llama.cpp, CPU**, KV reused across turns | F16 | 8 | 2.57 / 6.28 ¹ | 842 | 23.4 | 1.12 GB | 82% ¹ |
| **llama.cpp, CPU** | **Q8_0** | 8 | 2.14 / 5.50 | 685 | 17.4 | 0.60 GB | 79% |
| **llama.cpp, CPU** | **Q4_K_M** | 8 | 1.46 / 3.68 | 481 | 12.8 | 0.37 GB | 76% |
| llama.cpp, CPU | Q8_0 | 4 | 2.22 / 5.65 ¹ | 733 | 17.9 | 0.60 GB | 82% ¹ |
| llama.cpp, CPU | Q8_0 | 16 | 1.88 / 4.73 ¹ | 624 | 15.5 | 0.60 GB | 82% ¹ |

¹ first 30 or 50 of the 100 tasks.

Q8_0 on the other splits (8 threads, 100 tasks each):

| Split | Command P50 / P95 (s) | Success | Rule fast path: share of commands / their success |
|---|---:|---:|---:|
| test_id | 1.80 / 4.49 | 95% | 17% / 100% |
| test_expr | 2.14 / 5.50 | 79% | 6% / 100% |
| test_entity | 3.01 / 4.80 | 86% | 0% / — |
| test_combo | 4.12 / 5.86 | 87% | 0% / — |

- Switching from PyTorch eager to llama.cpp is the main win: the 0.6B model is bound by per-token overhead, not by
  the hardware (even the GPU runs at 32 ms / token in eager mode). llama.cpp also keeps the KV cache of every earlier
  turn, so each turn only processes the new tool result.
- Q8_0 is lossless here: on all 100 tasks it succeeds and fails on exactly the same tasks as BF16 on the GPU, at
  2.1 s per command on 8 CPU threads. On the same first 30 tasks it takes 2.2 s vs 12.6 s for PyTorch FP32 on the
  same CPU (5.7x faster), with identical success (25 / 30).
- Q4_K_M is another 31% faster but loses 3 of the 100 tasks (paired test p = 0.25, not significant at this size).
- Threads scale weakly (4 → 8 → 16): decoding a small model is memory-bound.
- The rule fast path is deliberately strict (one absolute setting, short sentence). It answers every command it takes
  correctly in milliseconds, but takes only 0–17% of them, so the median barely moves.
- A command still takes ~3.5 model turns of ~27 tokens each. Fewer turns and shorter tool calls are the next lever.

### Training cost

| Stage | Time on RTX 4070 Laptop |
|---|---|
| SFT, 3,000 expert trajectories, 1 epoch | 47 min |
| Hard-task mining, 400 prompts × 8 samples | 85 min |
| GRPO / DAPO, 40 steps × 4 prompts × 8 samples | ~1.7 h (~2.5 min / step) |
| DPO, 164 pairs | 36 min |

Two engineering fixes made this fit on one 8 GB GPU: batched multi-turn rollouts (4.7x faster sampling), and
releasing the generation KV cache between rollout and update. On Windows (WDDM), an over-full GPU spills into shared
system memory instead of raising OOM, which had silently slowed a step from 2.5 to 22 minutes.

## Quick start

```bash
pip install -r requirements.txt
python scripts/make_data.py                         # train / dev / test splits
python -m unittest discover -s tests -v
python -m cockpit.eval.evaluate --policy expert     # 100% on every split by construction
python -m cockpit.eval.evaluate --policy rule       # rule-NLU baseline
```

Training (needs `models/Qwen3-0.6B`, e.g. `modelscope download --model Qwen/Qwen3-0.6B --local_dir models/Qwen3-0.6B`):

```bash
python -m cockpit.train.sft --config configs/sft_qwen3_0.6b.json
bash scripts/run_rl_pipeline.sh                     # hard-task mining -> GRPO -> DPO -> evaluation (resumable)
python scripts/stats.py                             # intervals and paired tests
bash scripts/run_parallel_json_pipeline.sh          # parallel calls + efficiency reward (resumable)
python scripts/efficiency_stats.py                  # efficiency tables
```

On-device deployment (llama.cpp, CPU):

```bash
pip install llama-cpp-python gguf sentencepiece
python scripts/export_merged.py --lora runs/grpo_v2/lora_step40.pt --out runs/export/grpo_v2
python <llama.cpp>/convert_hf_to_gguf.py runs/export/grpo_v2 --outtype q8_0 --outfile grpo_v2-q8_0.gguf
python scripts/bench_device.py --gguf grpo_v2-q8_0.gguf --threads 8 --tag llama_q8_t8
```

## Layout

```
cockpit/
  spec.py          shared business rules (temperature range, what "a bit" means, call budget)
  env/             vehicle state, POI / media catalog, tools, episode
  verifier.py      private verifier
  tasks/           templates, task generator, splits, optional LLM paraphrasing
  agent/           token-level chat format, episode runner, batched runner
  baselines/       scripted expert, rule NLU
  train/           LoRA, SFT, GRPO / DAPO, DPO, reward
  eval/            evaluation harness
  serve/           vLLM client, router, MCP server
  rag/             owner's-manual retrieval (BM25)
scripts/           data, mining, pipeline, failure analysis, statistics, latency
results/           result summaries used in this README
```

## License

MIT
