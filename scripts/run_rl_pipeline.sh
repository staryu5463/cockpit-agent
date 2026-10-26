#!/usr/bin/env bash
# Resumable RL pipeline: every step is skipped if its output already exists,
# so after an interruption just run it again.
#   bash scripts/run_rl_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
BASE=models/Qwen3-0.6B
SFT=runs/sft_qwen3_0.6b/lora.pt
SPLITS="test_id test_expr test_entity test_combo"
export PYTHONIOENCODING=utf-8

step() { echo; echo "=== $(date '+%F %T') $*"; }

# 1. mine hard tasks + DPO pairs from the SFT policy
if [ ! -s data/rl_hard.stats.json ]; then
  step "mine hard tasks"
  $PY scripts/mine_hard.py --model $BASE --lora $SFT --n-regular 200 --n-hard 200 --k 8 \
      --out-tasks data/rl_hard.jsonl --out-pairs data/pairs_r1.jsonl
fi

# 2. GRPO / DAPO on the mined tasks (resumes from the latest checkpoint)
if [ ! -s runs/grpo_v2/lora_step40.pt ]; then
  step "GRPO"
  $PY -m cockpit.train.grpo --config configs/grpo_qwen3_0.6b.json --override '{
    "output_dir": "runs/grpo_v2", "train_file": "data/rl_hard.jsonl", "tiers": null,
    "steps": 40, "ppo_epochs": 2, "lr": 2e-5, "max_new_tokens": 128, "micro_batch": 1,
    "max_resample": 3, "save_every": 5, "resume": true}'
fi

# 3. DPO on the mined preference pairs
if [ ! -s runs/dpo_v1/lora.pt ]; then
  step "DPO"
  $PY -m cockpit.train.dpo --config configs/dpo_qwen3_0.6b.json --override '{
    "pairs_file": "data/pairs_r1.jsonl", "output_dir": "runs/dpo_v1"}'
fi

# 4. evaluate + failure attribution
for NAME in grpo_v2 dpo_v1; do
  LORA=runs/$NAME/lora.pt; [ "$NAME" = grpo_v2 ] && LORA=runs/grpo_v2/lora_step40.pt
  if [ ! -s runs/eval/$NAME/summary.json ]; then
    step "eval $NAME"
    $PY -m cockpit.eval.evaluate --policy hf --model $BASE --lora $LORA --name $NAME --splits $SPLITS --batch-size 16
    for S in $SPLITS; do $PY scripts/analyze_failures.py runs/eval/$NAME --split $S --examples 0 > runs/eval/$NAME/failures_$S.txt; done
  fi
done
step "done"
