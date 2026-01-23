#!/bin/bash
# MedSeRL Self-Play Training Script
# Two-phase game: Injector modifies note → Assessor classifies

set -e

# Configuration
OUTPUT_DIR="outputs/self_play"
EXPERIMENT_NAME="medserl_selfplay_v2"

# Model (adjust as needed)
MODEL_PATH="Qwen/Qwen3-4B"

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
python3 scripts/self_play/preprocess_medec.py \
    --input data_processed/medec_paired/train_val_split/rl_train.jsonl \
    --output data_processed/self_play/train.parquet \
    --injection-prompts configs/prompts/error_injection_prompts_v2.json

# Also preprocess validation data if exists
VAL_FILE="data_processed/self_play/val.parquet"
if [ -f "data_processed/medec_paired/train_val_split/rl_val.jsonl" ]; then
    python3 scripts/self_play/preprocess_medec.py \
        --input data_processed/medec_paired/train_val_split/rl_val.jsonl \
        --output "$VAL_FILE" \
        --injection-prompts configs/prompts/error_injection_prompts_v2.json
else
    echo "Warning: No separate validation file, using training file for validation"
    VAL_FILE="data_processed/self_play/train.parquet"
fi

echo ""
echo "=== Step 2: Starting Self-Play Training ==="
echo "Using verl with multi-turn interaction"

# Step 2: Launch training with verl
# Multi-turn enabled for Injector → Assessor game
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files="$PROJECT_ROOT/data_processed/self_play/train.parquet" \
    data.val_files="$PROJECT_ROOT/$VAL_FILE" \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_remove_padding=False \
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
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$PROJECT_ROOT/scripts/self_play/configs/interaction_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=2 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    critic.optim.lr=1e-5 \
    critic.model.path=$MODEL_PATH \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.ppo_mini_batch_size=16 \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    custom_reward_function.path="$PROJECT_ROOT/scripts/self_play/reward_function.py" \
    custom_reward_function.name=compute_score \
    trainer.logger=console \
    trainer.project_name='medserl-selfplay' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=3 \
    trainer.save_freq=-1 \
    trainer.test_freq=100

echo ""
echo "=================================================="
echo "Training Complete!"
echo "=================================================="
echo "Outputs: $OUTPUT_DIR"
echo "Logs: results/self_play/interactions/"
echo "=================================================="