#!/bin/bash
# MedSeRL Training Script
# Runs training in a detached screen session

# Install screen if not available
if ! command -v screen &> /dev/null; then
    echo "Installing screen..."
    apt-get update && apt-get install -y screen
fi

SESSION_NAME="train_screen"

# Create screen session and run training
screen -dmS $SESSION_NAME bash -c '
cd /workspace
source miniconda3/bin/activate
cd med_serl
conda activate med_serl

# 2x RTX 6000 Pro optimized settings (SeRL-aligned: 100% self-instruction)
python scripts/train_gpu.py \
    --model_path google/medgemma-4b-it \
    --num_samples 512 \
    --sft_epochs 3 \
    --rl_episodes 50 \
    --batch_size 16 \
    --rl_lr 7e-7 \
    --use_self_instruction \
    --si_rouge_threshold 0.7 \
    --si_difficulty_lower 0.2 \
    --si_difficulty_upper 0.8

exec bash
'

echo "Training started in screen session: $SESSION_NAME"
echo "To attach: screen -r $SESSION_NAME"
echo "To detach: Ctrl+A, then D"
