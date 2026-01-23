#!/bin/bash
# MedSeRL verl Training Script
# Two-phase self-play training with Injector → Assessor game

set -e

# Configuration
PROJECT_ROOT="/Users/josmaiga/Documents/GitHub/med_serl"
DATA_DIR="$PROJECT_ROOT/data_processed/selfplay"
MODEL_PATH="google/medgemma-4b-it"
OUTPUT_DIR="$PROJECT_ROOT/outputs/verl_training"

# Training hyperparameters
TRAIN_BATCH_SIZE=512
RL_EPISODES=50
LEARNING_RATE=5e-7

echo "=================================================="
echo "MedSeRL verl Self-Play Training"
echo "=================================================="
echo "Project: $PROJECT_ROOT"
echo "Data: $DATA_DIR"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "=================================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Activate environment if needed
if [ -d "$PROJECT_ROOT/med_serl" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/med_serl/bin/activate"
fi

# Change to project directory
cd "$PROJECT_ROOT"

# Run verl training with multi-turn interaction enabled
python3 -m verl.trainer.main_ppo \
    --config-path="verl_implementation/config" \
    --config-name='ppo_trainer' \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$PROJECT_ROOT/verl_implementation/config/interaction_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=2 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='medserl-selfplay' \
    trainer.experiment_name='injector-assessor-game' \
    trainer.total_epochs=$RL_EPISODES \
    trainer.save_freq=5 \
    trainer.default_local_dir="$OUTPUT_DIR" \
    algorithm.adv_estimator=gae \
    algorithm.kl_ctrl.kl_coef=0.001 \
    critic.optim.lr=$LEARNING_RATE \
    actor.optim.lr=$LEARNING_RATE

echo ""
echo "=================================================="
echo "Training complete!"
echo "Outputs saved to: $OUTPUT_DIR"
echo "=================================================="
