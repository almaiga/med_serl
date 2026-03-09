#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# MedReRcT Detection + Localization — Background Launcher
# =============================================================================
# Launches scripts/medrect/inference_detection.py inside a detached screen.
#
# Usage:
#   bash scripts/medrect/run_medrect_bg.sh <model_name>
#
#   bash scripts/medrect/run_medrect_bg.sh medrect-sft
#   bash scripts/medrect/run_medrect_bg.sh Qwen3-8B
#   MAX_SAMPLES=50 bash scripts/medrect/run_medrect_bg.sh medrect-sft
#   NO_THINKING=1  bash scripts/medrect/run_medrect_bg.sh Qwen3-8B
#
# Optional env overrides:
#   DATASET         — ms | uw | all            (default: all)
#   BATCH_SIZE      — int                       (default: 8)
#   TEMPERATURE     — float                     (default: 0.7)
#   THINKING_BUDGET — int tokens for thinking   (default: 1024)
#   MAX_NEW_TOKENS  — int                       (default: 512)
#   MAX_SAMPLES     — cap for quick tests       (default: all)
#   NO_THINKING     — set to 1 to disable       (default: 0)
#   BASE_MODEL_PATH — local path to base model  (required for LoRA in offline env)
#
# Example (offline container, LoRA adapter):
#   BASE_MODEL_PATH=/workspace/models/Qwen3-8B \
#     bash scripts/medrect/run_medrect_bg.sh medrect-sft
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

INFERENCE_SCRIPT="scripts/medrect/inference_detection.py"
PROMPT_CONFIG="configs/prompts/detection_localization_prompts.json"

# ── Model registry ────────────────────────────────────────────────────────────
declare -A MODELS=(
    ["medrect-sft"]="outputs/local_training/qwen3-8b-medrect-sft"
    ["medrect-sft-4b"]="outputs/local_training/qwen3-4b-medrect-sft"
    ["HuatuoGPT-o1-7B"]="FreedomIntelligence/HuatuoGPT-o1-7B"
    ["Qwen3-4B"]="Qwen/Qwen3-4B"
    ["Qwen3-8B"]="Qwen/Qwen3-8B"
    ["Qwen3-14B"]="Qwen/Qwen3-14B"
    ["Qwen3-32B"]="Qwen/Qwen3-32B"
)

# ── Resolve model ─────────────────────────────────────────────────────────────
MODEL_NAME="${1:-}"
if [[ -z "${MODEL_NAME}" ]]; then
    echo "Usage: bash $0 <model_name>"
    echo ""
    echo "Available models:"
    for k in "${!MODELS[@]}"; do printf "  %-30s %s\n" "${k}" "${MODELS[$k]}"; done
    exit 1
fi

MODEL_PATH="${MODELS[${MODEL_NAME}]:-}"
if [[ -z "${MODEL_PATH}" ]]; then
    # Allow passing a raw path directly as the model name
    if [[ -d "${MODEL_NAME}" ]]; then
        MODEL_PATH="${MODEL_NAME}"
    else
        echo "ERROR: unknown model '${MODEL_NAME}'"
        echo "Add it to the MODELS table in this script, or pass a directory path directly."
        exit 1
    fi
fi

# ── Parameters ────────────────────────────────────────────────────────────────
DATASET="${DATASET:-all}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEMPERATURE="${TEMPERATURE:-0.7}"
THINKING_BUDGET="${THINKING_BUDGET:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
NO_THINKING="${NO_THINKING:-0}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/detection/${MODEL_NAME}}"

# ── Paths ─────────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${OUTPUT_DIR}/logs"
LOG_FILE="${LOG_DIR}/${DATASET}_${TIMESTAMP}.log"
SCREEN_NAME="medrect_${MODEL_NAME//\//_}"

# ── Validation ────────────────────────────────────────────────────────────────
[[ -f "${PROMPT_CONFIG}" ]] || { echo "ERROR: ${PROMPT_CONFIG} not found"; exit 1; }
[[ -f "${INFERENCE_SCRIPT}" ]] || { echo "ERROR: ${INFERENCE_SCRIPT} not found"; exit 1; }

# Warn if running LoRA adapter offline without BASE_MODEL_PATH
ADAPTER_CONFIG="${MODEL_PATH}/adapter_config.json"
if [[ -f "${ADAPTER_CONFIG}" && -z "${BASE_MODEL_PATH}" ]]; then
    echo "WARNING: '${MODEL_PATH}' looks like a LoRA adapter but BASE_MODEL_PATH is not set."
    echo "         If running offline, set: BASE_MODEL_PATH=/path/to/base/model"
    echo ""
fi

# ── Screen guard ──────────────────────────────────────────────────────────────
if ! command -v screen &>/dev/null; then
    echo "screen not found, installing..."
    command -v apt-get &>/dev/null && apt-get update -qq && apt-get install -y screen \
        || { echo "ERROR: install screen manually"; exit 1; }
fi

if screen -list 2>/dev/null | grep -q "${SCREEN_NAME}"; then
    echo "Screen session '${SCREEN_NAME}' is already running."
    echo "  Attach : screen -r ${SCREEN_NAME}"
    echo "  Kill   : screen -X -S ${SCREEN_NAME} quit"
    exit 1
fi

mkdir -p "${LOG_DIR}"

# ── Build command ─────────────────────────────────────────────────────────────
CMD="python -u ${INFERENCE_SCRIPT}"
CMD+=" --model_path ${MODEL_PATH}"
CMD+=" --prompt_config ${PROMPT_CONFIG}"
CMD+=" --dataset ${DATASET}"
CMD+=" --batch_size ${BATCH_SIZE}"
CMD+=" --temperature ${TEMPERATURE}"
CMD+=" --thinking_budget ${THINKING_BUDGET}"
CMD+=" --max_new_tokens ${MAX_NEW_TOKENS}"
CMD+=" --output_dir ${OUTPUT_DIR}"
[[ -n "${MAX_SAMPLES}"    ]] && CMD+=" --max_samples ${MAX_SAMPLES}"
[[ "${NO_THINKING}" == "1" ]] && CMD+=" --no_thinking"
[[ -n "${BASE_MODEL_PATH}" ]] && CMD+=" --base_model_path ${BASE_MODEL_PATH}"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  MedReRcT Detection + Localization Inference"
echo "============================================================"
echo "  Model      : ${MODEL_NAME}  →  ${MODEL_PATH}"
[[ -n "${BASE_MODEL_PATH}" ]] && echo "  Base model : ${BASE_MODEL_PATH}"
echo "  Dataset    : ${DATASET}"
echo "  Batch size : ${BATCH_SIZE}"
echo "  Temperature: ${TEMPERATURE}"
echo "  Thinking   : $([ "${NO_THINKING}" == "1" ] && echo "disabled" || echo "enabled (budget=${THINKING_BUDGET} tok)")"
echo "  Max tokens : ${MAX_NEW_TOKENS}"
echo "  Max samples: ${MAX_SAMPLES:-all}"
echo "  Output     : ${OUTPUT_DIR}"
echo "  Log        : ${LOG_FILE}"
echo "  Screen     : ${SCREEN_NAME}"
echo "============================================================"
command -v nvidia-smi &>/dev/null \
    && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
    || true
echo ""
echo "  $ HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ${CMD}"
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
FULL_CMD="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ${CMD} 2>&1 | tee -a '${LOG_FILE}'"
FULL_CMD+="; echo ''; echo \"=== DONE (exit \$?) ===\""

screen -dmS "${SCREEN_NAME}" bash -c "${FULL_CMD}"
sleep 1

echo "Launched in screen session '${SCREEN_NAME}'"
echo "  Attach : screen -r ${SCREEN_NAME}"
echo "  Detach : Ctrl-A, D"
echo "  Logs   : tail -f ${LOG_FILE}"
echo ""
