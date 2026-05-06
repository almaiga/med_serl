#!/usr/bin/env bash
set -eo pipefail

# =============================================================================
# 4-GPU parallel launcher — wraps run_detection.sh
#
# Usage:
#   bash scripts/detect_loc/run_detection_4gpu.sh <model_name> [thinking|no-thinking]
#
# Splits the dataset across 4 GPUs (one shard per GPU) and launches each
# shard in its own screen session. All other env overrides (DATASET,
# BATCH_SIZE, THINKING_BUDGET, MAX_SAMPLES, ...) are forwarded as-is.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/run_detection.sh"

MODEL_NAME="${1:-}"
MODE="${2:-}"

if [[ -z "${MODEL_NAME}" ]]; then
    echo "Usage: bash $0 <model_name> [thinking|no-thinking]"
    exit 1
fi

NUM_GPUS=4

echo ""
echo "============================================================"
echo "  4-GPU Parallel Launch"
echo "  Model : ${MODEL_NAME}  |  Mode : ${MODE:-default}"
echo "  Shards: ${NUM_GPUS}  (one per GPU)"
echo "============================================================"
echo ""

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Launching shard ${GPU_ID}/${NUM_GPUS} on GPU ${GPU_ID}..."
    GPU_ID="${GPU_ID}" \
    SHARD_ID="${GPU_ID}" \
    NUM_SHARDS="${NUM_GPUS}" \
        bash "${LAUNCHER}" "${MODEL_NAME}" "${MODE}"
    sleep 1
done

echo ""
echo "All ${NUM_GPUS} shards launched. To monitor:"
echo ""
echo "  screen -list | grep detect"
echo ""
echo "To follow logs:"
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    echo "  GPU ${GPU_ID}: tail -f \$(ls -t results/detection/*/logs/*shard${GPU_ID}*.log 2>/dev/null | head -1)"
done
echo ""
