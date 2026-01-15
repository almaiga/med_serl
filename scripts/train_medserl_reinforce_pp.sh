#!/bin/bash
#
# MedSeRL Training Script - REINFORCE++ with Inline VCF
#
# This script launches online RL training for MedSeRL with:
# - Batch-based training (32/64/128 notes per round)
# - Inline VCF filtering during injector rollouts
# - Zero-sum rewards for injector-assessor self-play
# - REINFORCE++ policy gradient algorithm
#
# Usage:
#   bash scripts/train_medserl_reinforce_pp.sh
#

set -e  # Exit on error
set -x  # Print commands

# ============================================================================
# Configuration
# ============================================================================

# Model paths
export PRETRAIN=${PRETRAIN:-"google/medgemma-4b-it"}
export REF_MODEL=${REF_MODEL:-$PRETRAIN}

# Data paths (raw MEDEC data - NO pre-filtering needed)
export PROMPT_DATA=${PROMPT_DATA:-"data/medec_train.jsonl"}
export PRETRAIN_DATA=${PRETRAIN_DATA:-"data/medec_test.jsonl"}

# Output paths
export SAVE_PATH=${SAVE_PATH:-"outputs/medserl_checkpoints"}
export LOGGING_DIR=${LOGGING_DIR:-"outputs/logs"}

# Training hyperparameters (REINFORCE++)
export ACTOR_LEARNING_RATE=${ACTOR_LEARNING_RATE:-5e-7}
export KL_COEF=${KL_COEF:-1e-4}
export ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR:-"reinforce"}

# Batch sizes
export MICRO_TRAIN_BATCH_SIZE=${MICRO_TRAIN_BATCH_SIZE:-2}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-64}

# Training schedule
export MAX_EPOCHS=${MAX_EPOCHS:-5}
export SAVE_STEPS=${SAVE_STEPS:-100}
export LOGGING_STEPS=${LOGGING_STEPS:-1}

# LoRA config
export LORA_RANK=${LORA_RANK:-16}
export LORA_ALPHA=${LORA_ALPHA:-32}

# VCF configuration (inline filtering during rollouts)
export VCF_MIN_JACCARD=${VCF_MIN_JACCARD:-0.85}
export VCF_MAX_JACCARD=${VCF_MAX_JACCARD:-0.99}
export VCF_MAX_WORD_EDITS=${VCF_MAX_WORD_EDITS:-6}
export VCF_MAX_RETRIES=${VCF_MAX_RETRIES:-3}

# vLLM Engine Configuration
export VLLM_NUM_ENGINES=${VLLM_NUM_ENGINES:-2}
export VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.4}

# Generation settings
export TEMPERATURE=${TEMPERATURE:-0.7}
export TOP_P=${TOP_P:-0.9}

# Distributed training
export NUM_NODES=${NUM_NODES:-1}
export NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-1}

# ============================================================================
# Pre-flight Checks
# ============================================================================

echo "=== MedSeRL REINFORCE++ Training ==="
echo ""
echo "Configuration:"
echo "  Model: $PRETRAIN"
echo "  Training Data: $PROMPT_DATA"
echo "  Output: $SAVE_PATH"
echo "  Batch Size: $ROLLOUT_BATCH_SIZE"
echo "  VCF: Jaccard [$VCF_MIN_JACCARD, $VCF_MAX_JACCARD], Max Edits $VCF_MAX_WORD_EDITS"
echo ""

# Check if Ray cluster is running
if ! command -v ray &> /dev/null; then
    echo "[ERROR] Ray not found. Please install Ray:"
    echo "  pip install ray"
    exit 1
fi

# Check if data files exist
if [ ! -f "$PROMPT_DATA" ]; then
    echo "[ERROR] Training data not found: $PROMPT_DATA"
    exit 1
fi

# Create output directories
mkdir -p "$SAVE_PATH"
mkdir -p "$LOGGING_DIR"

# ============================================================================
# Launch Training
# ============================================================================

echo ""
echo "Training Architecture: Batch-Based Online RL"
echo "  1. Sample batch of $ROLLOUT_BATCH_SIZE notes"
echo "  2. Injector rollout + VCF filtering (inline, max $VCF_MAX_RETRIES retries)"
echo "  3. Assessor rollout"
echo "  4. Compute zero-sum rewards"
echo "  5. Policy update (REINFORCE++)"
echo "  6. Repeat with new batch"
echo ""
echo "Starting training..."
echo ""

# Option 1: Using OpenRLHF's train_ppo_ray CLI (if available)
if command -v python3 -m openrlhf.cli.train_ppo_ray &> /dev/null; then
    echo "[INFO] Using OpenRLHF train_ppo_ray CLI"

    ray job submit \
        --address="http://127.0.0.1:8265" \
        --runtime-env-json='{"working_dir": "/workspace"}' \
        -- python3 -m openrlhf.cli.train_ppo_ray \
            --ref_num_nodes $NUM_NODES \
            --ref_num_gpus_per_node $NUM_GPUS_PER_NODE \
            --reward_pretrain $PRETRAIN \
            --pretrain_data $PRETRAIN_DATA \
            --prompt_data $PROMPT_DATA \
            --save_path $SAVE_PATH \
            --micro_train_batch_size $MICRO_TRAIN_BATCH_SIZE \
            --train_batch_size $TRAIN_BATCH_SIZE \
            --rollout_batch_size $ROLLOUT_BATCH_SIZE \
            --max_epochs $MAX_EPOCHS \
            --save_steps $SAVE_STEPS \
            --logging_steps $LOGGING_STEPS \
            --actor_learning_rate $ACTOR_LEARNING_RATE \
            --kl_coef $KL_COEF \
            --advantage_estimator $ADVANTAGE_ESTIMATOR \
            --lora_rank $LORA_RANK \
            --lora_alpha $LORA_ALPHA \
            --gradient_checkpointing \
            --normalize_reward \
            --vcf_min_jaccard $VCF_MIN_JACCARD \
            --vcf_max_jaccard $VCF_MAX_JACCARD \
            --vcf_max_word_edits $VCF_MAX_WORD_EDITS \
            --vcf_max_retries $VCF_MAX_RETRIES \
            --use_wandb

# Option 2: Using custom MedSeRL trainer
else
    echo "[INFO] Using custom MedSeRL trainer"

    python3 src/training/medserl_trainer.py \
        --pretrain $PRETRAIN \
        --ref_model $REF_MODEL \
        --prompt_data $PROMPT_DATA \
        --pretrain_data $PRETRAIN_DATA \
        --save_path $SAVE_PATH \
        --logging_dir $LOGGING_DIR \
        --micro_train_batch_size $MICRO_TRAIN_BATCH_SIZE \
        --train_batch_size $TRAIN_BATCH_SIZE \
        --rollout_batch_size $ROLLOUT_BATCH_SIZE \
        --max_epochs $MAX_EPOCHS \
        --actor_learning_rate $ACTOR_LEARNING_RATE \
        --kl_coef $KL_COEF \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --vcf_min_jaccard $VCF_MIN_JACCARD \
        --vcf_max_jaccard $VCF_MAX_JACCARD \
        --vcf_max_word_edits $VCF_MAX_WORD_EDITS \
        --vcf_max_retries $VCF_MAX_RETRIES \
        --interaction_log_path $LOGGING_DIR/interactions.jsonl \
        --metrics_log_path $LOGGING_DIR/metrics.jsonl
fi

echo ""
echo "=== Training Complete ==="
echo "Checkpoints: $SAVE_PATH"
echo "Logs: $LOGGING_DIR"
echo "Interactions: $LOGGING_DIR/interactions.jsonl"
echo "Metrics: $LOGGING_DIR/metrics.jsonl"
