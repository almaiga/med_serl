#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
TRAIN_FILE="${TRAIN_FILE:-data_processed/medrect/generated_assessor_all_sft.jsonl}"

OUTPUT_NAME="${OUTPUT_NAME:-qwen3-4b-assessor-sft}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/local_training/${OUTPUT_NAME}}"

SMOKE="${SMOKE:-0}"
LIMIT="${LIMIT:-}"
EVAL_SPLIT="${EVAL_SPLIT:-0.05}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"
NO_EARLY_STOPPING="${NO_EARLY_STOPPING:-0}"
WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-medrect-assessor-sft}"
DEBUG_SAMPLES="${DEBUG_SAMPLES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"

LOG_DIR="${LOG_DIR:-outputs/local_training/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${OUTPUT_NAME}_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
    echo "ERROR: training file not found: ${TRAIN_FILE}"
    exit 1
fi

if [[ "${SMOKE}" == "1" ]]; then
    LIMIT="${LIMIT:-64}"
fi

if [[ -z "${NUM_EPOCHS}" ]]; then
    if [[ "${SMOKE}" == "1" ]]; then
        NUM_EPOCHS="1"
    else
        NUM_EPOCHS="3"
    fi
fi

if [[ -z "${NPROC_PER_NODE}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NPROC_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
    else
        NPROC_PER_NODE="1"
    fi
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    LAUNCHER=(torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}")
else
    LAUNCHER=(python3)
fi

TRAIN_CMD=(
    "${LAUNCHER[@]}"
    scripts/medrect/train_medrect_lora.py
    --train-file "${TRAIN_FILE}"
    --model-name "${MODEL_NAME}"
    --output-dir "${OUTPUT_DIR}"
    --eval-split "${EVAL_SPLIT}"
    --max-seq-length "${MAX_SEQ_LENGTH}"
    --per-device-train-batch-size "${BATCH_SIZE}"
    --gradient-accumulation-steps "${GRAD_ACCUM}"
    --learning-rate "${LEARNING_RATE}"
    --num-train-epochs "${NUM_EPOCHS}"
    --lora-r "${LORA_R}"
    --lora-alpha "${LORA_ALPHA}"
    --lora-dropout "${LORA_DROPOUT}"
    --lora-target-modules "${LORA_TARGET_MODULES}"
    --debug-samples "${DEBUG_SAMPLES}"
    --bf16
)

if [[ -n "${LIMIT}" ]]; then
    TRAIN_CMD+=(--limit "${LIMIT}")
fi

if [[ "${NO_EARLY_STOPPING}" == "1" ]]; then
    TRAIN_CMD+=(--no-early-stopping)
fi

if [[ "${WANDB}" == "1" ]]; then
    TRAIN_CMD+=(--wandb --wandb-project "${WANDB_PROJECT}")
fi

echo "============================================================"
echo "  Assessor-Only SFT"
echo "============================================================"
echo "Model:            ${MODEL_NAME}"
echo "Train file:       ${TRAIN_FILE}"
echo "Output dir:       ${OUTPUT_DIR}"
echo "Log file:         ${LOG_FILE}"
echo "Smoke mode:       ${SMOKE}"
echo "Epochs:           ${NUM_EPOCHS}"
echo "Batch size:       ${BATCH_SIZE}"
echo "Grad accum:       ${GRAD_ACCUM}"
echo "Effective batch:  $((BATCH_SIZE * GRAD_ACCUM))"
echo "LoRA:             r=${LORA_R}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}"
echo "Launcher:         ${LAUNCHER[*]}"
echo "Wandb:            ${WANDB}"
echo "============================================================"
echo ""

"${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"

echo ""
echo "============================================================"
echo "Finished assessor-only SFT"
echo "Adapter dir: ${OUTPUT_DIR}"
echo "Log file:    ${LOG_FILE}"
echo "============================================================"
