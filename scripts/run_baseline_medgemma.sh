loop#!/bin/bash
# Run baseline MedGemma inference with few-shot ICL

DATASET=${1:-"ms"}
SESSION_NAME="baseline_medgemma"

# Kill existing session if running
screen -X -S $SESSION_NAME quit 2>/dev/null

echo "=========================================="
echo "🔬 Baseline MedGemma Inference (with ICL)"
echo "=========================================="
echo "Model: google/medgemma-4b-it"
echo "Dataset: $DATASET"
echo "Few-shot: Yes (ICL)"
echo "Session: $SESSION_NAME"
echo "=========================================="

screen -dmS $SESSION_NAME bash -c "
source /workspace/miniconda3/bin/activate
conda activate med_serl

python scripts/inference_error_detection.py \
    --model_path google/medgemma-4b-it \
    --model_name medgemma_baseline \
    --model_type medgemma \
    --dataset $DATASET \
    --temperature 0.3 \
    --max_new_tokens 512 \
    --output_dir results/inference

echo ''
echo '✅ Baseline inference completed!'
exec bash
"

echo ""
echo "Started in screen session: $SESSION_NAME"
echo "To monitor: screen -r $SESSION_NAME"
echo "To detach: Ctrl+A, then D"
