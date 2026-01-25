#!/bin/bash
# MedSeRL Self-Play Training Script
# Two-phase game: Injector modifies note → Assessor classifies

set -e

# Configuration
MODEL_PATH="Qwen/Qwen3-4B"
MODEL_SHORT=$(basename $MODEL_PATH)  # e.g., "Qwen3-4B"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="medserl_selfplay_${MODEL_SHORT}_${TIMESTAMP}"

# Output directory with model and date for tracking
OUTPUT_DIR="outputs/self_play/${MODEL_SHORT}_${TIMESTAMP}"

# Get absolute path to project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_DIR="$PROJECT_ROOT/scripts/self_play/configs"

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p results/self_play

echo "=================================================="
echo "MedSeRL Self-Play Training"
echo "=================================================="
echo "Project root: $PROJECT_ROOT"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "=================================================="

# Step 1: Preprocess data (generates 2 examples per pair)
echo ""
echo "=== Step 1: Preprocessing MEDEC data ==="
# Use v3 prompts that don't leak the answer to the model
python3 scripts/self_play/preprocess_medec.py \
    --input data_processed/medec_paired/train_val_split/rl_train.jsonl \
    --output data_processed/self_play/train.parquet \
    --injection-prompts configs/prompts/error_injection_prompts_v3.json \
    --max-pairs 10

# Also preprocess validation data if exists
VAL_FILE="data_processed/self_play/val.parquet"
if [ -f "data_processed/medec_paired/train_val_split/rl_val.jsonl" ]; then
    python3 scripts/self_play/preprocess_medec.py \
        --input data_processed/medec_paired/train_val_split/rl_val.jsonl \
        --output "$VAL_FILE" \
        --injection-prompts configs/prompts/error_injection_prompts_v3.json \
        --max-pairs 50
else
    echo "Warning: No separate validation file, using training file for validation"
    VAL_FILE="data_processed/self_play/train.parquet"
fi

echo ""
echo "=== Step 2: Starting Self-Play Training ==="
echo "Using verl for medical error detection RL training"
echo "Multi-turn with sglang - NOTE: sglang initialization can take 5-10 minutes"
echo ""

# Ensure PYTHONPATH includes project root for interaction imports
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"

# Step 2: Launch training with verl
# Two-turn training with multi-turn enabled:
# Turn 1: Model acts as Injector - modifies the note
# Turn 2: Model acts as Assessor - classifies the modified note
# MedicalGameInteraction orchestrates the turns and computes zero-sum rewards
#
# CRITICAL: Multi-turn requires sglang rollout, NOT vllm!
# See: https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html
#
# Using --config-path and --config-name to load base YAML config first,
# which defines the multi_turn nested structure that Hydra needs.
# Then we override specific values via CLI (following verl's examples).
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_DIR" \
    --config-name="ppo_multiturn" \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files="$PROJECT_ROOT/data_processed/self_play/train.parquet" \
    data.val_files="$PROJECT_ROOT/$VAL_FILE" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$CONFIG_DIR/interaction_config.yaml" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward_model.enable=False \
    custom_reward_function.path="$PROJECT_ROOT/scripts/self_play/reward_function.py" \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='medserl-selfplay' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=3 \
    trainer.save_freq=9999 \
    trainer.test_freq=10 \
    trainer.val_before_train=True

echo ""
echo "=================================================="
echo "Training Complete!"
echo "=================================================="
echo "Outputs: $OUTPUT_DIR"
echo "Logs: results/self_play/interactions/"
echo ""

# Step 3: Analyze training results
echo "=== Step 3: Analyzing Training Results ==="
if [ -d "results/self_play/interactions" ]; then
    python3 scripts/self_play/analyze_training.py \
        --log-dir results/self_play/interactions \
        --samples 3
fi

echo "=================================================="