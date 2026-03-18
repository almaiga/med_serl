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

JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-4B}"
JUDGE_PORT="${JUDGE_PORT:-8002}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

echo "=================================================="
echo "  MedSeRL Agentic UMLS Judge Server"
echo "  Model   : $JUDGE_MODEL"
echo "  Port    : $JUDGE_PORT"
echo "  GPU mem : $GPU_MEM_UTIL"
echo "=================================================="
echo ""
echo "  Once started, set this on the TRAINING pod:"
echo "    export JUDGE_VLLM_URL=http://$(hostname).runpod.internal:${JUDGE_PORT}/v1/chat/completions"
echo ""
echo "  (hostname = this pod's RunPod ID, reachable via Global Networking)"
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
