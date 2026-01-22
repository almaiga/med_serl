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
VAL_FILE="data_processed/self_play/val.parquet"
if [ -f "data_processed/medec_paired/train_val_split/rl_val.jsonl" ]; then
    python3 scripts/self_play/preprocess_medec.py \
        --input data_processed/medec_paired/train_val_split/rl_val.jsonl \
        --output "$VAL_FILE"
else
    echo "Warning: No separate validation file found, will use training file for validation"
    VAL_FILE="data_processed/self_play/train.parquet"
fi

echo "=== Starting Self-Play Training ==="
echo "Using verl default config with overrides"

# Step 2: Launch training with verl
# Using verl's default ppo_trainer.yaml config with command-line overrides
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files="$PROJECT_ROOT/data_processed/self_play/train.parquet" \
    data.val_files="$PROJECT_ROOT/$VAL_FILE" \
    data.train_batch_size=64 \
    data.max_prompt_length=256 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    critic.optim.lr=1e-5 \
    critic.model.path=$MODEL_PATH \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.ppo_mini_batch_size=16 \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=console \
    trainer.project_name='medserl-selfplay' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=3 \
    trainer.save_freq=500 \
    trainer.test_freq=100

echo "=== Training Complete ==="
echo "Outputs saved to: $OUTPUT_DIR"
echo "Game logs saved to: results/self_play/"