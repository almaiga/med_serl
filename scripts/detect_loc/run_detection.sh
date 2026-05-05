#!/usr/bin/env bash
set -eo pipefail

# =============================================================================
# Detection + Localization Inference — Launcher
# =============================================================================
# Models are cached at /workspace/.cache/huggingface — runs fully offline.
#
# Usage:
#   bash scripts/detect_loc/run_detection.sh <model_name> [thinking|no-thinking]
#
#   bash scripts/detect_loc/run_detection.sh Qwen3-8B no-thinking
#   bash scripts/detect_loc/run_detection.sh Qwen3-8B thinking
#   bash scripts/detect_loc/run_detection.sh medrect-assessor no-thinking
#   bash scripts/detect_loc/run_detection.sh medrect-sft no-thinking
#   MAX_SAMPLES=50 bash scripts/detect_loc/run_detection.sh Qwen3-8B no-thinking
#
# Optional env overrides:
#   DATASET         — ms | uw | all  (default: all)
#   BATCH_SIZE      — default: 8
#   MODE            — thinking | no-thinking  (default: no-thinking)
#   TEMPERATURE     — mode default: 0.6 thinking, 0.7 no-thinking
#   TOP_P           — mode default: 0.95 thinking, 0.8 no-thinking
#   TOP_K           — mode default: 20
#   MIN_P           — mode default: 0
#   PRESENCE_PENALTY— mode default: 0
#   THINKING_BUDGET — mode default: 32768 thinking
#   MAX_NEW_TOKENS  — mode default: 16384 no-thinking
#   MAX_SAMPLES     — limit samples  (default: all)
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
        medrect-r2|selfplay-r2|medserl-r2|Abdine/medserl-qwen3-4b-medrect-mixed-selfplay-r2)
                         echo "Abdine/medserl-qwen3-4b-medrect-mixed-selfplay-r2" ;;
        medrect-assessor|qwen3-4b-medrect-assessor|Abdine/qwen3-4b-medrect-assessor)
                         echo "Abdine/qwen3-4b-medrect-assessor" ;;
        medrect-sft)     echo "outputs/local_training/qwen3-8b-medrect-sft" ;;
        *)               echo "" ;;
    esac
}

model_size_slug() {
    case "$1" in
        *4[Bb]*)  echo "4b" ;;
        *8[Bb]*)  echo "8b" ;;
        *14[Bb]*) echo "14b" ;;
        *32[Bb]*) echo "32b" ;;
        *72[Bb]*) echo "72b" ;;
        *)        echo "unknown_size" ;;
    esac
}

# ── Resolve model name ──────────────────────────────────────────────────
MODEL_NAME="${1:-}"
MODE="${2:-${MODE:-}}"
if [[ -z "${MODEL_NAME}" ]]; then
    echo "Usage: bash $0 <model_name> [thinking|no-thinking]"
    echo ""
    echo "Available models:"
    echo "  HuatuoGPT-o1-7B            FreedomIntelligence/HuatuoGPT-o1-7B"
    echo "  Qwen3-4B                   Qwen/Qwen3-4B"
    echo "  Qwen3-8B                   Qwen/Qwen3-8B"
    echo "  Qwen3-14B                  Qwen/Qwen3-14B"
    echo "  Qwen3-32B                  Qwen/Qwen3-32B"
    echo "  medrect-mixed              Abdine/qwen3-4b-medrect-mixed"
    echo "  medrect-r2                 Abdine/medserl-qwen3-4b-medrect-mixed-selfplay-r2"
    echo "  medrect-assessor           Abdine/qwen3-4b-medrect-assessor"
    echo "  medrect-sft                outputs/local_training/qwen3-8b-medrect-sft"
    exit 1
fi

if [[ -z "${MODE}" ]]; then
    if [[ "${NO_THINKING:-1}" == "0" ]]; then
        MODE="thinking"
    else
        MODE="no-thinking"
    fi
fi

case "${MODE}" in
    thinking|think)
        MODE="thinking"
        NO_THINKING=0
        ;;
    no-thinking|non-thinking|nothinking|no_thinking|no-think|none)
        MODE="no-thinking"
        NO_THINKING=1
        ;;
    *)
        echo "ERROR: unknown mode '${MODE}'. Use 'thinking' or 'no-thinking'."
        exit 1
        ;;
esac

MODEL_PATH="$(resolve_model_path "${MODEL_NAME}")"
if [[ -z "${MODEL_PATH}" ]]; then
    # Allow direct local paths and HF repo ids in addition to registry aliases.
    MODEL_PATH="${MODEL_NAME}"
fi

MODEL_REF="${MODEL_PATH}"
MODEL_SLUG="${MODEL_REF//\//_}"
MODEL_SLUG="${MODEL_SLUG//[^A-Za-z0-9._-]/_}"
MODEL_SIZE="$(model_size_slug "${MODEL_REF}")"

# ── Inference parameters ─────────────────────────────────────────────────────
DATASET="${DATASET:-all}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0}"
if [[ "${NO_THINKING}" == "1" ]]; then
    TEMPERATURE="${TEMPERATURE:-0.7}"
    TOP_P="${TOP_P:-0.8}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
    THINKING_BUDGET="${THINKING_BUDGET:-32768}"
else
    TEMPERATURE="${TEMPERATURE:-0.6}"
    TOP_P="${TOP_P:-0.95}"
    THINKING_BUDGET="${THINKING_BUDGET:-32768}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
fi
PROMPT_CONFIG="configs/prompts/detection_localization_prompts.json"
OUTPUT_DIR="${OUTPUT_DIR:-results/detection/${MODEL_SIZE}_${MODEL_SLUG}}"

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
PYTHON_CMD="python ${INFERENCE_SCRIPT}"
PYTHON_CMD+=" --model_path ${MODEL_PATH}"
PYTHON_CMD+=" --prompt_config ${PROMPT_CONFIG}"
PYTHON_CMD+=" --dataset ${DATASET}"
PYTHON_CMD+=" --mode ${MODE}"
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
echo "Model size:    ${MODEL_SIZE}"
echo "Mode:          ${MODE}"
echo "HF cache:      ${HF_HOME}"
echo "Dataset:       ${DATASET}"
echo "Batch size:    ${BATCH_SIZE}"
echo "Temperature:   ${TEMPERATURE}"
echo "Top p / top k: ${TOP_P} / ${TOP_K}"
echo "Min p:         ${MIN_P}"
echo "Presence pen.: ${PRESENCE_PENALTY}"
echo "Thinking:      $([ "${NO_THINKING}" == "1" ] && echo disabled || echo "enabled (budget=${THINKING_BUDGET})")"
echo "Max new tok:   $([ "${NO_THINKING}" == "1" ] && echo "${MAX_NEW_TOKENS}" || echo "${THINKING_BUDGET}")"
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
