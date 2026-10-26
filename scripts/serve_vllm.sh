#!/usr/bin/env bash
# Serve a (merged) model with vLLM. Linux / WSL2 only.
#
#   bash scripts/serve_vllm.sh <model_dir> [variant]
#
# variants (run one at a time, then benchmark each with scripts/bench_latency.py):
#   base      no prefix caching                          -> baseline
#   prefix    automatic prefix caching                   -> long, repeated system prompt
#   fp8       prefix caching + FP8 weights (Ada / Hopper)
#   spec      prefix caching + speculative decoding (needs a draft model / EAGLE head)
#
# Flag names follow the vLLM docs at the time of writing; check `vllm serve --help`
# for the version you install, they have changed between releases.
set -euo pipefail
MODEL=${1:?model dir}
VARIANT=${2:-prefix}
COMMON=(--port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85 --served-model-name cockpit)

case "$VARIANT" in
  base)   EXTRA=(--no-enable-prefix-caching) ;;
  prefix) EXTRA=(--enable-prefix-caching) ;;
  fp8)    EXTRA=(--enable-prefix-caching --quantization fp8) ;;
  spec)   EXTRA=(--enable-prefix-caching
                 --speculative-config "{\"method\": \"eagle3\", \"model\": \"${DRAFT:?set DRAFT=<eagle head dir>}\", \"num_speculative_tokens\": 4}") ;;
  *) echo "unknown variant $VARIANT"; exit 1 ;;
esac

exec vllm serve "$MODEL" "${COMMON[@]}" "${EXTRA[@]}"
