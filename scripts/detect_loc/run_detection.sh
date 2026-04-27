#!/usr/bin/env bash
set -eo pipefail

# =============================================================================
# Detection + Localization Inference — Launcher
# =============================================================================
# Models are cached at /workspace/.cache/huggingface — runs fully offline.
#
# Usage:
#   bash scripts/detect_loc/run_detection.sh <model_name>
#
#   bash scripts/detect_loc/run_detection.sh Qwen3-8B
#   bash scripts/detect_loc/run_detection.sh medrect-sft
#   MAX_SAMPLES=50 bash scripts/detect_loc/run_detection.sh Qwen3-8B
#   NO_THINKING=1  bash scripts/detect_loc/run_detection.sh Qwen3-8B
#
# Optional env overrides:
#   DATASET         — ms | uw | all  (default: all)
#   BATCH_SIZE      — default: 8
#   TEMPERATURE     — default: 0.6 with thinking, 0.7 without thinking
#   TOP_P           — default: 0.95 with thinking, 0.8 without thinking
#   TOP_K           — default: 20
#   MIN_P           — default: 0
#   PRESENCE_PENALTY— default: 0
#   THINKING_BUDGET — default: 4096
#   MAX_NEW_TOKENS  — default: 4096
#   MAX_SAMPLES     — limit samples  (default: all)
#   NO_THINKING     — set 1 to disable thinking mode (default: 0)
#   HF_HOME         — HuggingFace cache dir (default: /workspace/.cache/huggingface)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# ── HuggingFace cache — use offline mode only if cache exists ─────────────────
if [[ -d "/workspace/.cache/huggingface/hub" ]]; then
    export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
else
    export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
fi

# ── Model registry (bash 3 compatible) ───────────────────────────────────────
resolve_model_path() {
    case "$1" in
        HuatuoGPT-o1-7B) echo "FreedomIntelligence/HuatuoGPT-o1-7B" ;;
        Qwen3-4B)        echo "Qwen/Qwen3-4B" ;;
        Qwen3-8B)        echo "Qwen/Qwen3-8B" ;;
        Qwen3-14B)       echo "Qwen/Qwen3-14B" ;;
        Qwen3-32B)       echo "Qwen/Qwen3-32B" ;;
        medrect-mixed|qwen3-4b-medrect-mixed|Abdine/qwen3-4b-medrect-mixed)
                         echo "Abdine/qwen3-4b-medrect-mixed" ;;
        medrect-sft)     echo "outputs/local_training/qwen3-8b-medrect-sft" ;;
        *)               echo "" ;;
    esac
}

# ── Resolve model name ──────────────────────────────────────────────────
MODEL_NAME="${1:-}"
if [[ -z "${MODEL_NAME}" ]]; then
    echo "Usage: bash $0 <model_name>"
    echo ""
    echo "Available models:"
    echo "  HuatuoGPT-o1-7B            FreedomIntelligence/HuatuoGPT-o1-7B"
    echo "  Qwen3-4B                   Qwen/Qwen3-4B"
    echo "  Qwen3-8B                   Qwen/Qwen3-8B"
    echo "  Qwen3-14B                  Qwen/Qwen3-14B"
    echo "  Qwen3-32B                  Qwen/Qwen3-32B"
    echo "  medrect-mixed              Abdine/qwen3-4b-medrect-mixed"
    echo "  medrect-sft                outputs/local_training/qwen3-8b-medrect-sft"
    exit 1
fi

MODEL_PATH="$(resolve_model_path "${MODEL_NAME}")"
if [[ -z "${MODEL_PATH}" ]]; then
    # Allow direct local paths and HF repo ids in addition to registry aliases.
    MODEL_PATH="${MODEL_NAME}"
fi

MODEL_SLUG="${MODEL_NAME//\//_}"

# ── Inference parameters ─────────────────────────────────────────────────────
DATASET="${DATASET:-all}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
NO_THINKING="${NO_THINKING:-0}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0}"
THINKING_BUDGET="${THINKING_BUDGET:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
if [[ "${NO_THINKING}" == "1" ]]; then
    TEMPERATURE="${TEMPERATURE:-0.7}"
    TOP_P="${TOP_P:-0.8}"
else
    TEMPERATURE="${TEMPERATURE:-0.6}"
    TOP_P="${TOP_P:-0.95}"
fi
PROMPT_CONFIG="configs/prompts/detection_localization_prompts.json"
OUTPUT_DIR="results/detection/${MODEL_SLUG}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${OUTPUT_DIR}/logs"
LOG_FILE="${LOG_DIR}/${DATASET}_${TIMESTAMP}.log"
SCREEN_NAME="detect_${MODEL_SLUG}"

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
PYTHON_CMD="accelerate launch --config_file configs/accelerate_config.yaml ${INFERENCE_SCRIPT}"
PYTHON_CMD+=" --model_path ${MODEL_PATH}"
PYTHON_CMD+=" --prompt_config ${PROMPT_CONFIG}"
PYTHON_CMD+=" --dataset ${DATASET}"
PYTHON_CMD+=" --batch_size ${BATCH_SIZE}"
PYTHON_CMD+=" --temperature ${TEMPERATURE}"
PYTHON_CMD+=" --top_p ${TOP_P}"
PYTHON_CMD+=" --top_k ${TOP_K}"
PYTHON_CMD+=" --min_p ${MIN_P}"
PYTHON_CMD+=" --presence_penalty ${PRESENCE_PENALTY}"
PYTHON_CMD+=" --thinking_budget ${THINKING_BUDGET}"
PYTHON_CMD+=" --max_new_tokens ${MAX_NEW_TOKENS}"
PYTHON_CMD+=" --output_dir ${OUTPUT_DIR}"
[[ -n "${MAX_SAMPLES}" ]] && PYTHON_CMD+=" --max_samples ${MAX_SAMPLES}"
[[ "${NO_THINKING}" == "1" ]] && PYTHON_CMD+=" --no_thinking"

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Detection + Localization Inference"
echo "============================================================"
echo "Model:         ${MODEL_NAME}  (${MODEL_PATH})"
echo "HF cache:      ${HF_HOME}"
echo "Dataset:       ${DATASET}"
echo "Batch size:    ${BATCH_SIZE}"
echo "Temperature:   ${TEMPERATURE}"
echo "Top p / top k: ${TOP_P} / ${TOP_K}"
echo "Min p:         ${MIN_P}"
echo "Presence pen.: ${PRESENCE_PENALTY}"
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
HF_ENV="HF_HOME=${HF_HOME}"
if [[ "${HF_HUB_OFFLINE:-0}" == "1" ]]; then
    HF_ENV+=" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
fi
FULL_CMD="${HF_ENV} ${PYTHON_CMD} 2>&1 | tee -a '${LOG_FILE}'; echo ''; echo '=== DONE (exit \$?) ===' "
screen -dmS "${SCREEN_NAME}" bash -c "${FULL_CMD}"
sleep 1
echo "Started: screen -r ${SCREEN_NAME}"
echo "Logs:    tail -f ${LOG_FILE}"
