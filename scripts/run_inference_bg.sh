#!/bin/bash
# Run inference in background screen session

MODEL_PATH=${1:-"outputs/gpu_training/rl/rl_final"}
MODEL_NAME=${2:-"rl_final"}
DATASET=${3:-"ms"}

SESSION_NAME="inference"

# Kill existing session if running
screen -X -S $SESSION_NAME quit 2>/dev/null

echo "=========================================="
echo "🔬 Medical Error Detection Inference"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Name: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Session: $SESSION_NAME"
echo "=========================================="

screen -dmS $SESSION_NAME bash -c "
python scripts/inference_error_detection.py \
    --model_path $MODEL_PATH \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --temperature 0.3 \
    --max_new_tokens 512 \
    --no_few_shot \
    --output_dir results/inference

echo ''
echo '✅ Inference completed!'
exec bash
"

echo ""
echo "Started in screen session: $SESSION_NAME"
echo "To monitor: screen -r $SESSION_NAME"
echo "To detach: Ctrl+A, then D"
