#!/bin/bash
# MedSeRL Self-Play Training Script
# Uses verl with REINFORCE++ for medical error detection game

set -e

# Configuration
CONFIG_FILE="scripts/self_play/configs/self_play.yaml"
OUTPUT_DIR="outputs/self_play"
EXPERIMENT_NAME="medserl_selfplay_v1"

# Model (adjust as needed)
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"

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

# Step 2: Launch training with verl
python3 -m verl.trainer.main_ppo \
    $CONFIG_FILE \
    algorithm.adv_estimator=reinforce_plus_plus \
    algorithm.kl_coef=0.01 \
    model.path=$MODEL_PATH \
    trainer.output_dir=$OUTPUT_DIR \
    trainer.experiment_name=$EXPERIMENT_NAME \
    reward.reward_fn_path=scripts/self_play/rewards/zero_sum_reward.py

echo "=== Training Complete ==="
echo "Outputs saved to: $OUTPUT_DIR"
echo "Game logs saved to: results/self_play/"
