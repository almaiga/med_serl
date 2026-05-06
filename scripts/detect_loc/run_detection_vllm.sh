#!/usr/bin/env bash
set -eo pipefail

# =============================================================================
# Detection + Localization Inference — vLLM launcher
# =============================================================================
# Same model registry and output layout as run_detection.sh, but uses the
# vLLM backend (continuous batching) for higher throughput.
#
# Usage:
#   bash scripts/detect_loc/run_detection_vllm.sh <model_name> [thinking|no-thinking]
#
#   bash scripts/detect_loc/run_detection_vllm.sh medrect-r2 thinking
#   bash scripts/detect_loc/run_detection_vllm.sh medrect-r2 no-thinking
#   MAX_SAMPLES=20 bash scripts/detect_loc/run_detection_vllm.sh medrect-r2 thinking
#
# Optional env overrides:
#   DATASET              — ms | uw | all  (default: all)
#   MODE                 — thinking | no-thinking  (default: no-thinking)
#   TEMPERATURE          — mode default: 0.6 thinking, 0.7 no-thinking
#   TOP_P                — mode default: 0.95 thinking, 0.8 no-thinking
#   TOP_K                — default: 20
#   MIN_P                — default: 0
#   PRESENCE_PENALTY     — default: 0
#   THINKING_BUDGET      — default: 4096 (thinking only)
#   MAX_NEW_TOKENS       — default: 4096 (no-thinking only)
#   TENSOR_PARALLEL_SIZE — number of GPUs for tensor parallelism (default: 1)
#   MAX_SAMPLES          — limit samples (default: all)
#   HF_HOME              — HuggingFace cache dir
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# ── HuggingFace cache ─────────────────────────────────────────────────────────
if [[ -d "/workspace/.cache/huggingface/hub" ]]; then
    export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
else
    export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
fi

# ── Model registry ────────────────────────────────────────────────────────────
resolve_model_path() {
    case "$1" in
        HuatuoGPT-o1-7B) echo "FreedomIntelligence/HuatuoGPT-o1-7B" ;;
        Qwen3-4B)        echo "Qwen/Qwen3-4B" ;;
        Qwen3-8B)        echo "Qwen/Qwen3-8B" ;;
        Qwen3-14B)       echo "Qwen/Qwen3-14B" ;;
        Qwen3-32B)       echo "Qwen/Qwen3-32B" ;;
        medrect-mixed|qwen3-4b-medrect-mixed|Abdine/qwen3-4b-medrect-mixed)
                         echo "Abdine/qwen3-4b-medrect-mixed" ;;
        medrect-assessor|qwen3-4b-medrect-assessor|Abdine/qwen3-4b-medrect-assessor)
                         echo "Abdine/qwen3-4b-medrect-assessor" ;;
        medrect-sft)     echo "outputs/local_training/qwen3-8b-medrect-sft" ;;
        medrect-r2|selfplay-r2|medserl-r2|Abdine/medserl-qwen3-4b-medrect-mixed-selfplay-r2)
                         echo "Abdine/medserl-qwen3-4b-medrect-mixed-selfplay-r2" ;;
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

# ── Resolve model ─────────────────────────────────────────────────────────────
MODEL_NAME="${1:-}"
MODE="${2:-${MODE:-}}"
if [[ -z "${MODEL_NAME}" ]]; then
    echo "Usage: bash $0 <model_name> [thinking|no-thinking]"
    echo ""
    echo "Available models:"
    echo "  Qwen3-4B                   Qwen/Qwen3-4B"
    echo "  Qwen3-8B                   Qwen/Qwen3-8B"
    echo "  Qwen3-14B                  Qwen/Qwen3-14B"
    echo "  Qwen3-32B                  Qwen/Qwen3-32B"
    echo "  medrect-mixed              Abdine/qwen3-4b-medrect-mixed"
    echo "  medrect-assessor           Abdine/qwen3-4b-medrect-assessor"
    echo "  medrect-r2                 Abdine/medserl-qwen3-4b-medrect-mixed-selfplay-r2"
    exit 1
fi

if [[ -z "${MODE}" ]]; then
    MODE="no-thinking"
fi

case "${MODE}" in
    thinking|think)
        MODE="thinking"; NO_THINKING=0 ;;
    no-thinking|non-thinking|nothinking|no_thinking|no-think|none)
        MODE="no-thinking"; NO_THINKING=1 ;;
    *)
        echo "ERROR: unknown mode '${MODE}'. Use 'thinking' or 'no-thinking'."
        exit 1 ;;
esac

MODEL_PATH="$(resolve_model_path "${MODEL_NAME}")"
[[ -z "${MODEL_PATH}" ]] && MODEL_PATH="${MODEL_NAME}"

MODEL_SLUG="${MODEL_PATH//\//_}"
MODEL_SLUG="${MODEL_SLUG//[^A-Za-z0-9._-]/_}"
MODEL_SIZE="$(model_size_slug "${MODEL_PATH}")"

# ── Parameters ────────────────────────────────────────────────────────────────
DATASET="${DATASET:-all}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

if [[ "${NO_THINKING}" == "1" ]]; then
    TEMPERATURE="${TEMPERATURE:-0.7}"
    TOP_P="${TOP_P:-0.8}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
    THINKING_BUDGET="${THINKING_BUDGET:-4096}"
else
    TEMPERATURE="${TEMPERATURE:-0.6}"
    TOP_P="${TOP_P:-0.95}"
    THINKING_BUDGET="${THINKING_BUDGET:-4096}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4352}"
fi

PROMPT_CONFIG="configs/prompts/detection_localization_prompts.json"
OUTPUT_DIR="${OUTPUT_DIR:-results/detection/${MODEL_SIZE}_${MODEL_SLUG}}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${OUTPUT_DIR}/logs"
LOG_FILE="${LOG_DIR}/${DATASET}_vllm_${TIMESTAMP}.log"
SCREEN_NAME="detect_vllm_${MODEL_SLUG}"

# ── Ensure screen ─────────────────────────────────────────────────────────────
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

# ── Guard duplicate session ───────────────────────────────────────────────────
if screen -list 2>/dev/null | grep -q "${SCREEN_NAME}"; then
    echo "Screen session '${SCREEN_NAME}' already running!"
    echo "  Attach : screen -r ${SCREEN_NAME}"
    echo "  Kill   : screen -X -S ${SCREEN_NAME} quit"
    exit 1
fi

# ── Validate ──────────────────────────────────────────────────────────────────
INFERENCE_SCRIPT="scripts/medrect/inference_detection_vllm.py"
[[ -f "${PROMPT_CONFIG}" ]]    || { echo "ERROR: ${PROMPT_CONFIG} not found";    exit 1; }
[[ -f "${INFERENCE_SCRIPT}" ]] || { echo "ERROR: ${INFERENCE_SCRIPT} not found"; exit 1; }
mkdir -p "${LOG_DIR}"

# ── Build command ─────────────────────────────────────────────────────────────
PYTHON_CMD="python ${INFERENCE_SCRIPT}"
PYTHON_CMD+=" --model_path ${MODEL_PATH}"
PYTHON_CMD+=" --prompt_config ${PROMPT_CONFIG}"
PYTHON_CMD+=" --dataset ${DATASET}"
PYTHON_CMD+=" --mode ${MODE}"
PYTHON_CMD+=" --temperature ${TEMPERATURE}"
PYTHON_CMD+=" --top_p ${TOP_P}"
PYTHON_CMD+=" --top_k ${TOP_K}"
PYTHON_CMD+=" --min_p ${MIN_P}"
PYTHON_CMD+=" --presence_penalty ${PRESENCE_PENALTY}"
PYTHON_CMD+=" --thinking_budget ${THINKING_BUDGET}"
PYTHON_CMD+=" --max_new_tokens ${MAX_NEW_TOKENS}"
PYTHON_CMD+=" --tensor_parallel_size ${TENSOR_PARALLEL_SIZE}"
PYTHON_CMD+=" --output_dir ${OUTPUT_DIR}"
[[ -n "${MAX_SAMPLES}" ]] && PYTHON_CMD+=" --max_samples ${MAX_SAMPLES}"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Detection + Localization Inference (vLLM)"
echo "============================================================"
echo "Model:         ${MODEL_NAME}  (${MODEL_PATH})"
echo "Model size:    ${MODEL_SIZE}"
echo "Mode:          ${MODE}"
echo "HF cache:      ${HF_HOME}"
echo "Dataset:       ${DATASET}"
echo "Temperature:   ${TEMPERATURE}"
echo "Top p / top k: ${TOP_P} / ${TOP_K}"
echo "Min p:         ${MIN_P}"
echo "Presence pen.: ${PRESENCE_PENALTY}"
echo "Thinking:      $([ "${NO_THINKING}" == "1" ] && echo disabled || echo "enabled (budget=${THINKING_BUDGET})")"
echo "Max tokens:    $([ "${NO_THINKING}" == "1" ] && echo "${MAX_NEW_TOKENS}" || echo "$((THINKING_BUDGET + 256))")"
echo "Tensor par.:   ${TENSOR_PARALLEL_SIZE} GPU(s)"
echo "Max samples:   ${MAX_SAMPLES:-all}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "Log:           ${LOG_FILE}"
echo "Screen:        ${SCREEN_NAME}"
echo "============================================================"
command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""
echo "  ${PYTHON_CMD}"
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
HF_ENV="HF_HOME=${HF_HOME}"
[[ "${HF_HUB_OFFLINE:-0}" == "1" ]] && HF_ENV+=" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
FULL_CMD="${HF_ENV} ${PYTHON_CMD} 2>&1 | tee -a '${LOG_FILE}'; echo ''; echo '=== DONE (exit \$?) ==='"
screen -dmS "${SCREEN_NAME}" bash -c "${FULL_CMD}"
sleep 1
echo "Started: screen -r ${SCREEN_NAME}"
echo "Logs:    tail -f ${LOG_FILE}"
