#!/usr/bin/env bash
# Efficiency experiments (resumable): native JSON calls + parallel tool calls.
#   1. SFT on expert data in the native JSON style; 30% of tasks use the parallel schedule
#   2. GRPO / DAPO with the v2 recipe, with a success-only turn penalty and without it
#   3. evaluation on the four test splits
#   bash scripts/run_parallel_json_pipeline.sh
# (The compact fn-style run, scripts/run_efficiency_pipeline.sh, lost 13 points on unseen
#  phrasings; Qwen3 was pretrained on JSON tool calls, so this run keeps that format.)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PYTHONIOENCODING=utf-8 COCKPIT_CALL_STYLE=json COCKPIT_PARALLEL=1
BASE=models/Qwen3-0.6B
SFT=runs/sft_js_par/lora.pt
step() { echo; echo "=== $(date '+%F %T') $*"; }

if [ ! -s data/rl_mix.jsonl ]; then
  step "task mix"
  $PY scripts/make_rl_mix.py --n-parallel 100 --out data/rl_mix.jsonl
fi

if [ ! -s $SFT ]; then
  step "SFT (native JSON style, 30% parallel)"
  $PY -m cockpit.train.sft --config configs/sft_qwen3_0.6b.json \
      --override '{"output_dir": "runs/sft_js_par", "parallel_share": 0.3}'
fi

evaluate() {   # name lora
  if [ ! -s runs/eval/$1/episodes_test_entity.jsonl ]; then
    step "eval $1"
    $PY -m cockpit.eval.evaluate --policy hf --model $BASE --lora $2 --name $1 \
        --splits test_id test_expr test_entity --batch-size 16
  fi
  if [ ! -s runs/eval/$1_combo/episodes_test_combo.jsonl ]; then
    $PY -m cockpit.eval.evaluate --policy hf --model $BASE --lora $2 --name $1_combo \
        --splits test_combo --batch-size 8
  fi
}

evaluate sft_js_par $SFT

grpo() {   # output_dir reward_weights seed
  if [ ! -s $1/lora_step40.pt ]; then
    step "GRPO $1"
    $PY -m cockpit.train.grpo --config configs/grpo_qwen3_0.6b.json --override "{
      \"output_dir\": \"$1\", \"init_lora\": \"$SFT\", \"train_file\": \"data/rl_mix.jsonl\", \"tiers\": null,
      \"steps\": 40, \"ppo_epochs\": 2, \"lr\": 2e-5, \"max_new_tokens\": 128, \"micro_batch\": 1,
      \"max_resample\": 3, \"save_every\": 5, \"resume\": true, \"seed\": $3, \"reward_weights\": $2}"
  fi
}

BASE_W='{"success": 1.0, "format": 0.1, "steps": 0.05, "side": 0.2, "turns": 0.0}'
EFF_W='{"success": 1.0, "format": 0.1, "steps": 0.05, "side": 0.2, "turns": 0.1, "eff_on_success": 1}'
grpo runs/grpo_js_eff "$EFF_W" 0
evaluate grpo_js_eff runs/grpo_js_eff/lora_step40.pt
grpo runs/grpo_js_base "$BASE_W" 0
evaluate grpo_js_base runs/grpo_js_base/lora_step40.pt
step "done"
