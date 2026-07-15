#!/usr/bin/env bash
# serve_judge.sh — start MedRECT-32B as a vLLM OpenAI server for the judge.
#
# The training path (judge_client.py) talks HTTP to this. Thinking is controlled
# per-request by judge_client (chat_template_kwargs), so this is a standard serve
# — no thinking flag needed here. Run this on the judge GPU/pod, then point the
# training pod at http://<this-host>:<port>/v1/chat/completions
#
# Usage (on the judge pod):
#   bash scripts/self_play/serve_judge.sh
#   JUDGE_PORT=8002 JUDGE_TP=1 bash scripts/self_play/serve_judge.sh

set -euo pipefail

MODEL="${JUDGE_MODEL:-pfnet/Preferred-MedRECT-32B}"
PORT="${JUDGE_PORT:-8002}"
TP="${JUDGE_TP:-1}"
GPU_UTIL="${JUDGE_GPU_UTIL:-0.9}"

echo "=========================================================="
echo " Judge server: $MODEL"
echo "   port=$PORT  tensor_parallel=$TP  gpu_util=$GPU_UTIL"
echo "   URL for training pod:"
echo "     http://<this-host>:${PORT}/v1/chat/completions"
echo "=========================================================="

vllm serve "$MODEL" \
    --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --tensor-parallel-size "$TP" \
    --trust-remote-code
