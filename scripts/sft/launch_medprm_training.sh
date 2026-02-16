#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Med-PRM LoRA SFT Training Launcher
# =============================================================================
# Launches training in a detached screen session for background execution.
#
# Usage:
#   ./scripts/sft/launch_medprm_training.sh [4b|8b] [output_name]
#
# Examples:
#   ./scripts/sft/launch_medprm_training.sh 4b          # Train Qwen3-4B
#   ./scripts/sft/launch_medprm_training.sh 8b          # Train Qwen3-8B
#   ./scripts/sft/launch_medprm_training.sh 4b my_run   # Custom output name
# =============================================================================

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Parse arguments
MODEL_SIZE="${1:-4b}"
OUTPUT_NAME="${2:-qwen3-${MODEL_SIZE}-medprm-lora}"
SCREEN_NAME="medprm_train_${MODEL_SIZE}"

# Validate model size
if [[ "${MODEL_SIZE}" != "4b" && "${MODEL_SIZE}" != "8b" ]]; then
    echo "❌ Invalid model size: ${MODEL_SIZE}"
    echo "   Usage: $0 [4b|8b] [output_name]"
    exit 1
fi

# Training parameters (optimized for ~2400 samples)
LEARNING_RATE="1e-4"
NUM_EPOCHS="3"
LORA_R="16"
LORA_ALPHA="32"
LORA_DROPOUT="0.05"
EVAL_SPLIT="0.05"
EARLY_STOPPING_PATIENCE="3"

# Output directory
OUTPUT_DIR="outputs/local_training/${OUTPUT_NAME}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="outputs/local_training/logs/${OUTPUT_NAME}_${TIMESTAMP}.log"

# Create directories
mkdir -p "$(dirname "${LOG_FILE}")"
mkdir -p "${OUTPUT_DIR}"

# Check if screen is installed
if ! command -v screen &> /dev/null; then
    echo "📦 Installing screen..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y screen
    elif command -v brew &> /dev/null; then
        brew install screen
    else
        echo "❌ Cannot install screen. Please install manually."
        exit 1
    fi
fi

# Check if session already exists
if screen -list | grep -q "${SCREEN_NAME}"; then
    echo "⚠️  Screen session '${SCREEN_NAME}' already exists!"
    echo "   To attach: screen -r ${SCREEN_NAME}"
    echo "   To kill:   screen -X -S ${SCREEN_NAME} quit"
    exit 1
fi

# Activate virtual environment if exists
VENV_ACTIVATE=""
if [[ -f ".venv/bin/activate" ]]; then
    VENV_ACTIVATE="source .venv/bin/activate && "
elif [[ -f "venv/bin/activate" ]]; then
    VENV_ACTIVATE="source venv/bin/activate && "
fi

# Build training command
TRAIN_CMD="${VENV_ACTIVATE}python scripts/sft/train_medprm_lora.py \\
    --model-size ${MODEL_SIZE} \\
    --output-dir ${OUTPUT_DIR} \\
    --learning-rate ${LEARNING_RATE} \\
    --num-train-epochs ${NUM_EPOCHS} \\
    --lora-r ${LORA_R} \\
    --lora-alpha ${LORA_ALPHA} \\
    --lora-dropout ${LORA_DROPOUT} \\
    --eval-split ${EVAL_SPLIT} \\
    --early-stopping-patience ${EARLY_STOPPING_PATIENCE} \\
    --bf16 \\
    --debug-samples 2 \\
    2>&1 | tee ${LOG_FILE}"

# Print configuration
echo ""
echo "============================================================"
echo "🏥 Med-PRM LoRA SFT Training"
echo "============================================================"
echo "Model:          Qwen3-${MODEL_SIZE^^}"
echo "Output:         ${OUTPUT_DIR}"
echo "Log file:       ${LOG_FILE}"
echo "Screen session: ${SCREEN_NAME}"
echo ""
echo "Hyperparameters:"
echo "  Learning rate:  ${LEARNING_RATE}"
echo "  Epochs:         ${NUM_EPOCHS}"
echo "  LoRA r:         ${LORA_R}"
echo "  LoRA alpha:     ${LORA_ALPHA}"
echo "  LoRA dropout:   ${LORA_DROPOUT}"
echo "  Eval split:     ${EVAL_SPLIT}"
echo "  Early stopping: patience=${EARLY_STOPPING_PATIENCE}"
echo "============================================================"
echo ""

# Launch in screen
echo "🚀 Launching training in screen session '${SCREEN_NAME}'..."
screen -dmS "${SCREEN_NAME}" bash -c "${TRAIN_CMD}; exec bash"

sleep 2

# Verify screen started
if screen -list | grep -q "${SCREEN_NAME}"; then
    echo ""
    echo "✅ Training launched successfully!"
    echo ""
    echo "📋 Useful commands:"
    echo "   Attach to session:  screen -r ${SCREEN_NAME}"
    echo "   Detach from screen: Ctrl+A, then D"
    echo "   View log file:      tail -f ${LOG_FILE}"
    echo "   Kill session:       screen -X -S ${SCREEN_NAME} quit"
    echo "   List sessions:      screen -ls"
    echo ""
else
    echo "❌ Failed to start screen session"
    exit 1
fi
