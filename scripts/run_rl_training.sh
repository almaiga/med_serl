#!/bin/bash
# MedSeRL RL Training Script
# Runs OpenRLHF training in a detached screen session
#
# Usage:
#   bash scripts/run_rl_training.sh              # Full training
#   QUICK_TEST=1 bash scripts/run_rl_training.sh # Quick test

# Install screen if not available
if ! command -v screen &> /dev/null; then
    echo "Installing screen..."
    apt-get update && apt-get install -y screen
fi

SESSION_NAME="medserl_train"

# Export env vars for the screen session
export QUICK_TEST="${QUICK_TEST:-0}"
export MODEL_PATH="${MODEL_PATH:-google/medgemma-4b-it}"

# Create screen session and run training
screen -dmS $SESSION_NAME bash -c '
cd /workspace
source miniconda3/bin/activate
cd med_serl
conda activate med_serl

# Prepare data if not exists
if [ ! -f "data_processed/medec/seed_500.jsonl" ]; then
    echo "Preparing MEDEC data..."
    python scripts/prepare_medec_for_openrlhf.py
fi

# Run OpenRLHF training (SeRL-style, no SFT)
bash scripts/train_medserl_2gpu.sh

exec bash
'

echo "============================================"
echo "MedSeRL RL Training started!"
echo "============================================"
echo "Session: $SESSION_NAME"
echo "Quick test: $QUICK_TEST"
echo "Model: $MODEL_PATH"
echo ""
echo "To attach: screen -r $SESSION_NAME"
echo "To detach: Ctrl+A, then D"
echo "To kill:   screen -X -S $SESSION_NAME quit"
echo "============================================"
