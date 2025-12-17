#!/bin/bash
# Generate Post-Fill Reasoning Data using Qwen3-4B
# This script generates reasoning by showing the model the answer first
# Runs in a detached screen session

set -e

# Install screen if not available
if ! command -v screen &> /dev/null; then
    echo "Installing screen..."
    sudo apt-get update && sudo apt-get install -y screen
fi

MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
MODEL_NAME="qwen3-3b"
OUTPUT_DIR="data_processed/medec"
TEMPERATURE=0.7
MAX_TOKENS=256
BATCH_SIZE=1
SESSION_NAME="postfill_generation"

# Check if testing mode
if [ "$1" = "test" ]; then
    echo "🧪 Running in TEST mode (50 samples)"
    MAX_SAMPLES="--max_samples 50"
    OUTPUT_NAME="--output_name test_postfill_${MODEL_NAME}.jsonl"
else
    echo "🚀 Running FULL generation"
    MAX_SAMPLES=""
    OUTPUT_NAME="--output_name train_postfill_${MODEL_NAME}.jsonl"
fi

echo "=========================================="
echo "🔬 Generating Post-Fill Reasoning"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Temperature: $TEMPERATURE"
echo "Max Tokens: $MAX_TOKENS"
echo "Batch Size: $BATCH_SIZE"
echo "Output: $OUTPUT_DIR"
echo "Session: $SESSION_NAME"
echo "=========================================="
echo ""

# Create screen session and run generation
screen -dmS $SESSION_NAME bash -c "
cd $(pwd)

echo '=========================================='
echo '🔬 Generating Post-Fill Reasoning'
echo '=========================================='
echo 'Model: $MODEL_PATH'
echo 'Temperature: $TEMPERATURE'
echo '=========================================='
echo ''

python scripts/generate_postfill_reasoning.py \\
    --model_path \"$MODEL_PATH\" \\
    --model_name \"$MODEL_NAME\" \\
    --temperature $TEMPERATURE \\
    --max_new_tokens $MAX_TOKENS \\
    --batch_size $BATCH_SIZE \\
    --output_dir \"$OUTPUT_DIR\" \\
    $MAX_SAMPLES \\
    $OUTPUT_NAME

echo ''
echo '✅ Generation complete!'
echo 'Results saved to: $OUTPUT_DIR'

exec bash
"

echo "Generation started in screen session: $SESSION_NAME"
echo "To attach: screen -r $SESSION_NAME"
echo "To detach: Ctrl+A, then D"
echo "To list sessions: screen -ls"
