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

python scripts/train_gpu.py \
    --model_path google/medgemma-4b-it \
    --num_samples 512 \
    --sft_epochs 3 \
    --rl_episodes 50 \
    --batch_size 12 \
    --rl_lr 5e-7 \
    --use_self_instruction \
    --self_instruction_ratio 0.34

exec bash
'

echo "Training started in screen session: $SESSION_NAME"
echo "To attach: screen -r $SESSION_NAME"
echo "To detach: Ctrl+A, then D"
