#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# MedRECT-Style Mixed SFT Launcher
# =============================================================================
# Trains Qwen3-4B on the existing MedRECT-style mixed-role corpus:
#   - recovered assessor SFT (+ UW backfill)
#   - injector error DeepSeek chains
#   - injector benign DeepSeek chains
#
# Defaults:
#   - LoRA fine-tuning on Qwen/Qwen3-4B
#   - saves adapter locally
#   - can optionally merge to a full model directory
#   - can optionally upload adapter or merged model to Hugging Face
#
# Usage:
#   bash scripts/medrect/run_medrect_mixed_sft.sh
#
#   SMOKE=1 bash scripts/medrect/run_medrect_mixed_sft.sh
#
#   OUTPUT_NAME=qwen3-4b-medrect-mixed-v1 \
#   bash scripts/medrect/run_medrect_mixed_sft.sh
#
#   MERGE_FULL_MODEL=1 \
#   bash scripts/medrect/run_medrect_mixed_sft.sh
#
#   MERGE_FULL_MODEL=1 \
#   UPLOAD_TO_HF=1 \
#   UPLOAD_ARTIFACT=merged \
#   HF_REPO_ID=username/qwen3-4b-medrect-mixed-v1 \
#   bash scripts/medrect/run_medrect_mixed_sft.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
TRAIN_FILES="${TRAIN_FILES:-data_processed/medrect/assessor_sft_recovered_plus_uw.jsonl,data_processed/medrect/injector_error_chains_20260310_135156.jsonl,data_processed/medrect/injector_benign_chains_20260310_135156.jsonl}"

OUTPUT_NAME="${OUTPUT_NAME:-qwen3-4b-medrect-mixed-sft}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/local_training/${OUTPUT_NAME}}"
MERGED_OUTPUT_DIR="${MERGED_OUTPUT_DIR:-outputs/local_training/${OUTPUT_NAME}-merged}"

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
WANDB_PROJECT="${WANDB_PROJECT:-medrect-sft}"
DEBUG_SAMPLES="${DEBUG_SAMPLES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"

MERGE_FULL_MODEL="${MERGE_FULL_MODEL:-0}"
UPLOAD_TO_HF="${UPLOAD_TO_HF:-0}"
UPLOAD_ARTIFACT="${UPLOAD_ARTIFACT:-adapter}"  # adapter | merged
HF_REPO_ID="${HF_REPO_ID:-}"
HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

LOG_DIR="${LOG_DIR:-outputs/local_training/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${OUTPUT_NAME}_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

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

if [[ "${UPLOAD_ARTIFACT}" == "merged" && "${MERGE_FULL_MODEL}" != "1" ]]; then
    MERGE_FULL_MODEL="1"
fi

IFS=',' read -r -a TRAIN_FILE_ARRAY <<< "${TRAIN_FILES}"
for train_file in "${TRAIN_FILE_ARRAY[@]}"; do
    if [[ ! -f "${train_file}" ]]; then
        echo "ERROR: training file not found: ${train_file}"
        exit 1
    fi
done

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
    LAUNCHER=(python)
fi

TRAIN_CMD=(
    "${LAUNCHER[@]}"
    scripts/medrect/train_medrect_lora.py
    --train-file "${TRAIN_FILES}"
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
echo "  MedRECT Mixed SFT"
echo "============================================================"
echo "Model:            ${MODEL_NAME}"
echo "Train files:      ${TRAIN_FILES}"
echo "Output dir:       ${OUTPUT_DIR}"
echo "Merged output:    ${MERGED_OUTPUT_DIR}"
echo "Log file:         ${LOG_FILE}"
echo "Smoke mode:       ${SMOKE}"
echo "Epochs:           ${NUM_EPOCHS}"
echo "Batch size:       ${BATCH_SIZE}"
echo "Grad accum:       ${GRAD_ACCUM}"
echo "Effective batch:  $((BATCH_SIZE * GRAD_ACCUM))"
echo "LoRA:             r=${LORA_R}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}"
echo "Launcher:         ${LAUNCHER[*]}"
echo "Merge after train:${MERGE_FULL_MODEL}"
echo "Upload to HF:     ${UPLOAD_TO_HF}"
echo "Upload artifact:  ${UPLOAD_ARTIFACT}"
echo "============================================================"
echo ""

"${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"

if [[ "${MERGE_FULL_MODEL}" == "1" ]]; then
    echo ""
    echo "Merging adapter into full model..."
    python scripts/medrect/merge_medrect_lora.py \
        --adapter-dir "${OUTPUT_DIR}" \
        --output-dir "${MERGED_OUTPUT_DIR}" \
        --base-model "${MODEL_NAME}"
fi

if [[ "${UPLOAD_TO_HF}" == "1" ]]; then
    if [[ -z "${HF_REPO_ID}" ]]; then
        echo "ERROR: HF_REPO_ID must be set when UPLOAD_TO_HF=1"
        exit 1
    fi

    if [[ "${UPLOAD_ARTIFACT}" == "merged" ]]; then
        UPLOAD_DIR="${MERGED_OUTPUT_DIR}"
    else
        UPLOAD_DIR="${OUTPUT_DIR}"
    fi

    if [[ ! -d "${UPLOAD_DIR}" ]]; then
        echo "ERROR: upload directory not found: ${UPLOAD_DIR}"
        exit 1
    fi

    echo ""
    echo "Uploading ${UPLOAD_DIR} -> ${HF_REPO_ID}"
    python - "${HF_REPO_ID}" "${UPLOAD_DIR}" <<'PY'
import os
import sys

from huggingface_hub import HfApi, create_repo

repo_id = sys.argv[1]
folder_path = sys.argv[2]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)
api = HfApi(token=token)
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=folder_path,
)
print(f"Uploaded {folder_path} to {repo_id}")
PY
fi

echo ""
echo "============================================================"
echo "Finished"
echo "============================================================"
echo "Adapter: ${OUTPUT_DIR}"
if [[ "${MERGE_FULL_MODEL}" == "1" ]]; then
    echo "Merged:  ${MERGED_OUTPUT_DIR}"
fi
if [[ "${UPLOAD_TO_HF}" == "1" ]]; then
    echo "HF repo: ${HF_REPO_ID}"
fi
