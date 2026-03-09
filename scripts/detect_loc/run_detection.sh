#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Detection + Localization Inference — RunPod Launcher
# =============================================================================
# Each model lands in its own folder: results/detection/<model_name>/
#
# Usage:
#   bash scripts/detect_loc/run_detection.sh <model_name>
#
#   bash scripts/detect_loc/run_detection.sh HuatuoGPT-o1-7B
#   bash scripts/detect_loc/run_detection.sh Qwen3-8B
#   bash scripts/detect_loc/run_detection.sh medrect-sft
#   MAX_SAMPLES=50 bash scripts/detect_loc/run_detection.sh Qwen3-8B
#   NO_THINKING=1  bash scripts/detect_loc/run_detection.sh HuatuoGPT-o1-7B
#
# Optional env overrides:
#   DATASET         — ms | uw | all       (default: all)
#   BATCH_SIZE      — default: 8
#   TEMPERATURE     — default: 0.7
#   THINKING_BUDGET — default: 1024
#   MAX_NEW_TOKENS  — default: 512
#   MAX_SAMPLES     — limit samples       (default: all)
#   NO_THINKING     — set 1 to disable thinking mode
#   BASE_MODEL_PATH — local path to base model weights; required when running
#                     a LoRA adapter in an offline/air-gapped environment
#                     (the HF hub ID stored in adapter_config.json cannot be
#                     resolved offline)
#
# Model type detection (automatic):
#   • Local directory + adapter_config.json → LoRA adapter  (offline, needs BASE_MODEL_PATH)
#   • Local directory, no adapter_config    → merged/full local model (offline)
#   • No local directory (e.g. Qwen/Qwen3-8B) → HuggingFace hub model (online)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Model registry — add new models here ─────────────────────────────────────
declare -A MODELS=(
    ["HuatuoGPT-o1-7B"]="FreedomIntelligence/HuatuoGPT-o1-7B"
    ["Qwen3-4B"]="Qwen/Qwen3-4B"
    ["Qwen3-8B"]="Qwen/Qwen3-8B"
    ["Qwen3-14B"]="Qwen/Qwen3-14B"
    ["Qwen3-32B"]="Qwen/Qwen3-32B"
    ["medrect-sft"]="outputs/local_training/qwen3-8b-medrect-sft"
)

# ── Resolve model name ──────────────────────────────────────────────────
MODEL_NAME="${1:-}"
if [[ -z "${MODEL_NAME}" ]]; then
    echo "Usage: bash $0 <model_name>"
    echo ""
    echo "Available models:"
    for k in "${!MODELS[@]}"; do printf "  %-25s %s\n" "${k}" "${MODELS[$k]}"; done
    exit 1
fi

MODEL_PATH="${MODELS[${MODEL_NAME}]:-}"
if [[ -z "${MODEL_PATH}" ]]; then
    echo "ERROR: unknown model '${MODEL_NAME}'. Add it to the MODELS table in this script."
    exit 1
fi

# ── Detect model type ─────────────────────────────────────────────────────────
# IS_LOCAL=1  → local directory on disk (run fully offline)
# IS_LORA=1   → local directory that is a PEFT/LoRA adapter
# IS_LOCAL=0  → HuggingFace hub ID (needs internet, no offline flags)
IS_LOCAL=0
IS_LORA=0
HF_ENV=""

if [[ -d "${MODEL_PATH}" ]]; then
    IS_LOCAL=1
    HF_ENV="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
    if [[ -f "${MODEL_PATH}/adapter_config.json" ]]; then
        IS_LORA=1
    fi
fi

# ── Inference parameters (standard defaults) ───────────────────────────────
DATASET="${DATASET:-all}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEMPERATURE="${TEMPERATURE:-0.7}"
THINKING_BUDGET="${THINKING_BUDGET:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
NO_THINKING="${NO_THINKING:-0}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-}"   # local path to base model weights (LoRA offline)

# Warn if LoRA adapter but no base model override provided
if [[ "${IS_LORA}" == "1" && -z "${BASE_MODEL_PATH}" ]]; then
    echo "WARNING: '${MODEL_PATH}' is a LoRA adapter but BASE_MODEL_PATH is not set."
    echo "         The base model ID in adapter_config.json will be used; this will"
    echo "         FAIL in an offline environment. Set BASE_MODEL_PATH to a local copy."
    echo ""
fi
PROMPT_CONFIG="configs/prompts/detection_localization_prompts.json"
OUTPUT_DIR="results/detection/${MODEL_NAME}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${OUTPUT_DIR}/logs"
LOG_FILE="${LOG_DIR}/${DATASET}_${TIMESTAMP}.log"
SCREEN_NAME="detect_${MODEL_NAME}"

# ── Ensure screen is installed ──────────────────────────────────────────────
if ! command -v screen &>/dev/null; then
    echo "screen not found — installing..."
    if command -v apt-get &>/dev/null; then
        apt-get update -qq && apt-get install -y screen
    elif command -v brew &>/dev/null; then
        brew install screen
    else
        echo "ERROR: cannot install screen automatically."; exit 1
    fi
fi

# ── Guard against duplicate session ──────────────────────────────────────────
if screen -list 2>/dev/null | grep -q "${SCREEN_NAME}"; then
    echo "Screen session '${SCREEN_NAME}' already running!"
    echo "  Attach : screen -r ${SCREEN_NAME}"
    echo "  Kill   : screen -X -S ${SCREEN_NAME} quit"
    exit 1
fi

# ── Validate ──────────────────────────────────────────────────────────────
INFERENCE_SCRIPT="scripts/medrect/inference_detection.py"
[[ -f "${PROMPT_CONFIG}" ]] || { echo "ERROR: ${PROMPT_CONFIG} not found"; exit 1; }
[[ -f "${INFERENCE_SCRIPT}" ]] || { echo "ERROR: ${INFERENCE_SCRIPT} not found"; exit 1; }
mkdir -p "${LOG_DIR}"

# ── Build command ────────────────────────────────────────────────────────────
PYTHON_CMD="python -u ${INFERENCE_SCRIPT}"
PYTHON_CMD+=" --model_path ${MODEL_PATH}"
PYTHON_CMD+=" --prompt_config ${PROMPT_CONFIG}"
PYTHON_CMD+=" --dataset ${DATASET}"
PYTHON_CMD+=" --batch_size ${BATCH_SIZE}"
PYTHON_CMD+=" --temperature ${TEMPERATURE}"
PYTHON_CMD+=" --thinking_budget ${THINKING_BUDGET}"
PYTHON_CMD+=" --max_new_tokens ${MAX_NEW_TOKENS}"
PYTHON_CMD+=" --output_dir ${OUTPUT_DIR}"
[[ -n "${MAX_SAMPLES}" ]] && PYTHON_CMD+=" --max_samples ${MAX_SAMPLES}"
[[ "${NO_THINKING}" == "1" ]] && PYTHON_CMD+=" --no_thinking"
# Pass --base_model_path only for LoRA adapters
[[ "${IS_LORA}" == "1" && -n "${BASE_MODEL_PATH}" ]] && PYTHON_CMD+=" --base_model_path ${BASE_MODEL_PATH}"

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Detection + Localization Inference"
echo "============================================================"
if [[ "${IS_LORA}" == "1" ]]; then
    echo "Model:         ${MODEL_NAME}  (LoRA adapter → ${MODEL_PATH})"
    [[ -n "${BASE_MODEL_PATH}" ]] && echo "Base model:    ${BASE_MODEL_PATH}" \
        || echo "Base model:    (from adapter_config.json — online lookup)"
elif [[ "${IS_LOCAL}" == "1" ]]; then
    echo "Model:         ${MODEL_NAME}  (local → ${MODEL_PATH})"
else
    echo "Model:         ${MODEL_NAME}  (HuggingFace hub → ${MODEL_PATH})"
fi
echo "Dataset:       ${DATASET}"
echo "Batch size:    ${BATCH_SIZE}"
echo "Temperature:   ${TEMPERATURE}"
echo "Thinking:      $([ "${NO_THINKING}" == "1" ] && echo disabled || echo "enabled (budget=${THINKING_BUDGET})")"
echo "Max new tok:   ${MAX_NEW_TOKENS}"
echo "Max samples:   ${MAX_SAMPLES:-all}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "Log:           ${LOG_FILE}"
echo "Screen:        ${SCREEN_NAME}"
echo "============================================================"
command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""
echo "  ${PYTHON_CMD}"
echo ""

# ── Launch ───────────────────────────────────────────────────────────────────
FULL_CMD="${HF_ENV} ${PYTHON_CMD} 2>&1 | tee -a '${LOG_FILE}'; echo ''; echo '=== DONE (exit \$?) ===' "
screen -dmS "${SCREEN_NAME}" bash -c "${FULL_CMD}"
sleep 1
echo "Started: screen -r ${SCREEN_NAME}"
echo "Logs:    tail -f ${LOG_FILE}"
