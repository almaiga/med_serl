#!/bin/bash
# MedSeRL Training Script with GRPO Algorithm
#
# Alternative to REINFORCE++ using Group Relative Policy Optimization (GRPO)
# GRPO often provides better sample efficiency and stability.
#
# Usage:
#   bash run_medserl_grpo.sh [--model MODEL] [--gpus NUM_GPUS]

set -e

# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
NUM_GPUS="${NUM_GPUS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-64}"
GROUP_SIZE="${GROUP_SIZE:-8}"  # GRPO group size (n responses per prompt)
MAX_STEPS="${MAX_STEPS:-1000}"
LR="${LR:-1e-6}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --batch)
            TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --group)
            GROUP_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${DATA_DIR:-$HOME/data/medec}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/../outputs/verl_grpo}"
REWARD_PATH="${REWARD_PATH:-$PROJECT_DIR/reward/medec_reward.py}"

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "MedSeRL GRPO Training"
echo "============================================"
echo "Model:           $MODEL_PATH"
echo "GPUs:            $NUM_GPUS"
echo "Group Size:      $GROUP_SIZE"
echo "Batch Size:      $TRAIN_BATCH_SIZE"
echo "Learning Rate:   $LR"
echo "============================================"

# Check prerequisites
if [ ! -f "$DATA_DIR/medec_train.parquet" ]; then
    echo "Error: Training data not found. Run preprocess_medec.py first."
    exit 1
fi

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))

# Data files
TRAIN_FILES="$DATA_DIR/medec_train.parquet"
VAL_FILES="$DATA_DIR/medec_val.parquet"

# Run GRPO training
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=$GROUP_SIZE \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    \
    custom_reward_function.path="$REWARD_PATH" \
    custom_reward_function.name=compute_score \
    \
    trainer.total_epochs=3 \
    trainer.project_name=medserl \
    trainer.experiment_name=grpo \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    \
    +trainer.val_before_train=True \
    +trainer.n_gpus_per_node=$NUM_GPUS

echo "============================================"
echo "GRPO Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "============================================"
