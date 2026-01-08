#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for Qwen3-4B LoRA SFT.
# Adjust paths as needed on your SSH machine.

TRAIN_FILE="${TRAIN_FILE:-data_processed/medec_cot/sft_cot_training_data.jsonl}"
EVAL_FILE="${EVAL_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen3-4b-lora}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"

EXTRA_ARGS=()
if [[ -n "${EVAL_FILE}" ]]; then
  EXTRA_ARGS+=(--eval-file "${EVAL_FILE}")
fi

python scripts/sft/train_qwen3_4b_sft_lora.py \
  --train-file "${TRAIN_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-name "${MODEL_NAME}" \
  --bf16 \
  --max-seq-length 4096 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4 \
  --num-train-epochs 1 \
  --logging-steps 10 \
  --save-steps 200 \
  "${EXTRA_ARGS[@]}"
