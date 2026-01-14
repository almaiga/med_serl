#!/bin/bash
# MedSeRL Inference Script
# Runs inference in a detached screen session

set -e

# Install screen if not available
if ! command -v screen &> /dev/null; then
    echo "Installing screen..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install screen 2>/dev/null || echo "Please install screen: brew install screen"
    else
        apt-get update && apt-get install -y screen
    fi
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_path> [model_name] [extra_args...]"
    echo ""
    echo "Examples:"
    echo "  $0 Qwen/Qwen3-4B"
    echo "  $0 google/medgemma-4b-it medgemma"
    echo "  $0 google/medgemma-4b-it medgemma --dataset ms --max_samples 50"
    echo "  $0 google/medgemma-4b-it medgemma --batch_size 8"
    echo "  $0 trainer_output/qwen3-4b-medical-selfplay-sft sft_model --batch_size 4"
    echo ""
    echo "Supported models:"
    echo "  - Qwen3 models (Qwen/Qwen3-4B, etc.)"
    echo "  - MedGemma models (google/medgemma-4b-it, google/medgemma-4b-pt)"
    echo ""
    echo "Common options:"
    echo "  --batch_size N       Process N samples in parallel (default: 1)"
    echo "  --dataset [ms|uw|all]  Which test dataset to use (default: all)"
    echo "  --max_samples N      Limit to N samples"
    echo "  --temperature T      Sampling temperature (default: 0.7)"
    exit 1
fi

MODEL_PATH=$1
MODEL_NAME=${2:-$(basename $MODEL_PATH)}

# Shift past the first two arguments
shift
if [ $# -gt 0 ] && [[ ! "$1" =~ ^-- ]]; then
    shift
fi

# Collect remaining args
EXTRA_ARGS="$@"

SESSION_NAME="inference_screen"

echo "=========================================="
echo "🔬 Medical Error Detection Inference"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Name: $MODEL_NAME"
echo "Session: $SESSION_NAME"
echo "=========================================="
echo ""

# Create screen session and run inference
screen -dmS $SESSION_NAME bash -c "
cd $(pwd)

echo '=========================================='
echo '🔬 Medical Error Detection Inference'
echo '=========================================='
echo 'Model: $MODEL_PATH'
echo 'Name: $MODEL_NAME'
echo '=========================================='
echo ''

python scripts/inference_error_detection.py \\
    --model_path \"$MODEL_PATH\" \\
    --model_name \"$MODEL_NAME\" \\
    --dataset all \\
    --temperature 0 \\
    --max_new_tokens 256 \\
    --no_cot \\
    --batch_size 1 \\
    --output_dir results/inference \\
    $EXTRA_ARGS

echo ''
echo '✅ Inference completed!'
echo 'Results saved to: results/inference/'

exec bash
"

echo "Inference started in screen session: $SESSION_NAME"
echo "To attach: screen -r $SESSION_NAME"
echo "To detach: Ctrl+A, then D"
