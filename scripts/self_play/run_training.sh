#!/bin/bash
# MedSeRL Self-Play Training Script
# Uses verl with REINFORCE++ for medical error detection game

set -e

# Configuration
OUTPUT_DIR="outputs/self_play"
EXPERIMENT_NAME="medserl_selfplay_v1"

# Model (adjust as needed)
MODEL_PATH="Qwen/Qwen3-4B"

# Get absolute path to project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_DIR="$PROJECT_ROOT/scripts/self_play/configs"

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p results/self_play

# Step 1: Preprocess data
echo "=== Preprocessing MEDEC data ==="
python3 scripts/self_play/preprocess_medec.py \
    --input data_processed/medec_paired/train_val_split/rl_train.jsonl \
    --output data_processed/self_play/train.parquet

# Also preprocess validation data if exists
if [ -f "data_processed/medec_paired/train_val_split/rl_val.jsonl" ]; then
    python3 scripts/self_play/preprocess_medec.py \
        --input data_processed/medec_paired/train_val_split/rl_val.jsonl \
        --output data_processed/self_play/val.parquet
fi

echo "=== Starting Self-Play Training ==="
echo "Using config directory: $CONFIG_DIR"

# Step 2: Launch training with verl
python3 -m verl.trainer.main_ppo \
    --config-path "$CONFIG_DIR" \
    --config-name self_play \
    actor_rollout_ref.model.path=$MODEL_PATH \
    critic.model.path=$MODEL_PATH

echo "=== Training Complete ==="
echo "Outputs saved to: $OUTPUT_DIR"
echo "Game logs saved to: results/self_play/"