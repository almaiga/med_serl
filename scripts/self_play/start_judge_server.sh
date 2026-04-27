#!/bin/bash
# Start vLLM OpenAI-compatible server for the current Qwen judge.
#
# Run this on a SEPARATE RunPod instance (not the RL training pod).
# The training pod calls this server via JUDGE_VLLM_URL.
#
# Usage:
#   bash scripts/self_play/start_judge_server.sh
#
# Env overrides:
#   JUDGE_MODEL   — model to serve (default: Qwen/Qwen3-4B)
#   JUDGE_PORT    — port to listen on (default: 8002)
#   GPU_MEM_UTIL  — vLLM gpu_memory_utilization (default: 0.85)
#   JUDGE_HOST    — explicit host or pod ID to advertise to the training pod

SCREEN_SESSION="${SCREEN_SESSION:-medserl_judge_server}"
AUTO_SCREEN="${AUTO_SCREEN:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCREEN_LOG_DIR="${SCREEN_LOG_DIR:-${REPO_ROOT}/logs/screen}"

if [ "$AUTO_SCREEN" = "1" ] && [ -z "${STY:-}" ]; then
    if ! command -v screen >/dev/null 2>&1; then
        echo "ERROR: 'screen' is required but not installed."
        exit 1
    fi

    if screen -list | grep -q "\.${SCREEN_SESSION}[[:space:]]"; then
        echo "Screen session '${SCREEN_SESSION}' already exists."
        echo "Attach with: screen -r ${SCREEN_SESSION}"
        echo "Kill with:   screen -X -S ${SCREEN_SESSION} quit"
        exit 1
    fi

    mkdir -p "${SCREEN_LOG_DIR}"
    LOG_TS="$(date +%Y%m%d_%H%M%S)"
    SCREEN_LOG_FILE="${SCREEN_LOG_DIR}/${SCREEN_SESSION}_${LOG_TS}.log"
    SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
    QUOTED_ARGS=""
    for arg in "$@"; do
        QUOTED_ARGS+=" $(printf '%q' "$arg")"
    done

    echo "Launching ${SCRIPT_PATH} in screen session '${SCREEN_SESSION}'..."
    screen -L -Logfile "${SCREEN_LOG_FILE}" -dmS "${SCREEN_SESSION}" bash -lc "AUTO_SCREEN=0 bash $(printf '%q' "$SCRIPT_PATH")${QUOTED_ARGS}"
    echo "Attach with: screen -r ${SCREEN_SESSION}"
    echo "Kill with:   screen -X -S ${SCREEN_SESSION} quit"
    echo "Log file:    ${SCREEN_LOG_FILE}"
    exit 0
fi

JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-8B}"
JUDGE_PORT="${JUDGE_PORT:-8002}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

# Prefer a real RunPod pod identifier if the environment provides one.
# Fallback to hostname, but warn because in some containers hostname is only
# a short Docker/container ID and is not reachable via *.runpod.internal.
RAW_JUDGE_HOST="${JUDGE_HOST:-${RUNPOD_POD_ID:-${RUNPOD_ID:-${RUNPOD_POD_HOSTNAME:-$(hostname)}}}}"
if [[ "$RAW_JUDGE_HOST" == *.runpod.internal ]]; then
    JUDGE_ADVERTISED_URL="http://${RAW_JUDGE_HOST}:${JUDGE_PORT}/v1/chat/completions"
else
    JUDGE_ADVERTISED_URL="http://${RAW_JUDGE_HOST}.runpod.internal:${JUDGE_PORT}/v1/chat/completions"
fi

echo "=================================================="
echo "  MedSeRL Judge Server"
echo "  Model   : $JUDGE_MODEL"
echo "  Port    : $JUDGE_PORT"
echo "  GPU mem : $GPU_MEM_UTIL"
echo "=================================================="
echo ""
echo "  Once started, set this on the TRAINING pod:"
echo "    export JUDGE_VLLM_URL=${JUDGE_ADVERTISED_URL}"
echo ""
echo "  Host source: ${RAW_JUDGE_HOST}"
if [[ "${RAW_JUDGE_HOST}" =~ ^[0-9a-f]{12}$ ]]; then
    echo "  WARNING: ${RAW_JUDGE_HOST} looks like a container hostname, not a RunPod pod ID."
    echo "  If training cannot reach the judge, rerun with:"
    echo "    export JUDGE_HOST=<actual-runpod-pod-id>"
fi
echo "  (The advertised host must be reachable via RunPod Global Networking.)"
echo "  Both pods must have Global Networking enabled at deploy time."
echo ""
echo "  Then test with a direct HTTP call or launch:"
echo "    bash scripts/self_play/run_multiturn_training.sh"
echo ""
echo "=================================================="

python3 -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" \
    --port "$JUDGE_PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --served-model-name "$JUDGE_MODEL"
