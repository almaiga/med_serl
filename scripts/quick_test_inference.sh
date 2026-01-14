#!/bin/bash
# Quick Test Inference Script
# Tests inference on a small subset of data with batch processing

set -e

MODEL_PATH=${1:-"Qwen/Qwen3-4B"}
BATCH_SIZE=${2:-8}
MAX_SAMPLES=${3:-64}

echo "=========================================="
echo "🧪 Quick Inference Test"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Batch Size: $BATCH_SIZE"
echo "Max Samples: $MAX_SAMPLES"
echo "Dataset: all (MS + UW)"
echo "=========================================="
echo ""

python scripts/inference_error_detection.py \
    --model_path "$MODEL_PATH" \
    --model_name "$(basename $MODEL_PATH)" \
    --dataset all \
    --max_samples $MAX_SAMPLES \
    --batch_size $BATCH_SIZE \
    --temperature 0 \
    --max_new_tokens 256 \
    --no_cot \
    --output_dir results/inference

echo ""
echo "✅ Quick test completed!"
echo ""
echo "Results structure:"
echo "  results/inference/$(basename $MODEL_PATH)/no_cot/"
echo ""
echo "To view results:"
echo "  cat results/inference/$(basename $MODEL_PATH)/no_cot/*_summary.json | jq"
