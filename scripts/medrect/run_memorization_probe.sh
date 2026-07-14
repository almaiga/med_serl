#!/usr/bin/env bash
# Memorization probe: run each model on MEDEC test WITH and WITHOUT thinking.
#
# Logic:
#   thinking >> no-thinking  -> the model REASONS to solve the task
#   thinking ~= no-thinking  -> the model PATTERN-MATCHES / may have memorized
#                               (MEDEC derives from public exam content)
#
# Also answers: does the untrained base beat our trained models, and is that
# because it reasons or because it recalls?
#
# 4 models x 2 modes = 8 evals, ~5 min each (~40 min total).
# Deterministic (vLLM seed=0), --skip-existing safe to re-run.

set -euo pipefail
cd "$(dirname "$0")/../.."

TP="${TENSOR_PARALLEL_SIZE:-1}"
OUT_ROOT="${OUT_ROOT:-results/memorization_probe}"
DATASET="${DATASET:-all}"

# model_key -> HF path (base models + our two key trained models)
MODELS=(
  "base4b:Qwen/Qwen3-4B"
  "base8b:Qwen/Qwen3-8B"
  "sft_v2:Abdine/qwen3-4b-medrect-mixed-v2"
  "selfplay_step66:Abdine/qwen3-4b-medserl-v6-step66"
)

for entry in "${MODELS[@]}"; do
  key="${entry%%:*}"
  path="${entry#*:}"
  for mode in thinking no-thinking; do
    out_dir="${OUT_ROOT}/${key}__${mode}"
    if [[ -f "${out_dir}/DONE" ]]; then
      echo "== SKIP ${key} / ${mode} (already done) =="
      continue
    fi
    echo
    echo "=================================================================="
    echo " ${key}  |  mode=${mode}  |  ${path}"
    echo "=================================================================="
    python3 scripts/medrect/inference_detection_vllm.py \
      --model_path "${path}" \
      --dataset "${DATASET}" \
      --mode "${mode}" \
      --tensor_parallel_size "${TP}" \
      --output_dir "${out_dir}"
    touch "${out_dir}/DONE"
  done
done

echo
echo "=================================================================="
echo " Aggregating memorization matrix"
echo "=================================================================="
python3 scripts/medrect/aggregate_memorization_probe.py --results-root "${OUT_ROOT}"
