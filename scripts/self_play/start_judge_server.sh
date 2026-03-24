#!/bin/bash
# Start vLLM OpenAI-compatible server for the Agentic UMLS Judge.
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
echo "  MedSeRL Agentic UMLS Judge Server"
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
echo "  Then test with:"
echo "    bash scripts/self_play/run_judge_test.sh"
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
