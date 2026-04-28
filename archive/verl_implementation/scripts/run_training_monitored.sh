#!/bin/bash
# Enhanced MedSeRL VERL Training Script with Monitoring
# Follows Hugging Face GRPO blog best practices
# 
# Features:
# - WandB logging for experiment tracking
# - GPU monitoring telemetry
# - Automatic checkpointing and recovery
# - Optimized hyperparameters

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT="/Users/josmaiga/Documents/GitHub/med_serl"
DATA_DIR="$PROJECT_ROOT/data_processed/selfplay"
MODEL_PATH="google/medgemma-4b-it"
OUTPUT_DIR="$PROJECT_ROOT/outputs/verl_training"

# Training hyperparameters
TRAIN_BATCH_SIZE=512
RL_EPISODES=50
LEARNING_RATE=5e-7

# GPU Configuration (adjust for your hardware)
N_GPUS_PER_NODE=1  # Change to your GPU count
TENSOR_PARALLEL_SIZE=1  # Data parallelism (1) vs Model parallelism (>1)
ROLLOUT_GPU_MEMORY_UTIL=0.8  # 0-1, higher = more VRAM for inference

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

echo "=================================================="
echo "MedSeRL VERL Self-Play Training (Enhanced)"
echo "=================================================="
echo ""
echo "📋 Configuration:"
echo "   Project: $PROJECT_ROOT"
echo "   Data: $DATA_DIR"
echo "   Model: $MODEL_PATH"
echo "   Output: $OUTPUT_DIR"
echo "   GPUs per Node: $N_GPUS_PER_NODE"
echo "   Tensor Parallelism: $TENSOR_PARALLEL_SIZE"
echo "   GPU Memory Util: $ROLLOUT_GPU_MEMORY_UTIL"
echo ""

# Check if data files exist
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "❌ ERROR: Training data not found at $DATA_DIR/train.parquet"
    echo "   Run: python verl_implementation/data/preprocess_selfplay.py"
    exit 1
fi

# Check WandB authentication
echo "🔐 Checking WandB authentication..."
if [ ! -f ~/.netrc ] || ! grep -q "wandb" ~/.netrc; then
    echo "⚠️  WARNING: WandB not authenticated!"
    echo "   Run: wandb login"
    echo "   This must be done BEFORE training (saves 5+ mins of GPU time)"
    echo ""
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Activate environment if needed
if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo "📦 Activating virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
else
    echo "⚠️  No venv found, using system Python"
fi

# Change to project directory
cd "$PROJECT_ROOT"

# ============================================================================
# TRAINING START
# ============================================================================

echo ""
echo "🚀 Starting VERL training..."
echo "📊 Monitor GPU with: nvitop  (in another terminal)"
echo "📈 View WandB dashboard: https://wandb.ai/home"
echo ""

# Run VERL training with enhanced monitoring
python3 -m verl.trainer.main_ppo \
    --config-path="verl_implementation/config" \
    --config-name='ppo_trainer' \
    \
    # DATA CONFIGURATION
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.return_raw_chat=True \
    \
    # MODEL CONFIGURATION
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$PROJECT_ROOT/verl_implementation/config/interaction_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=2 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    \
    # GPU OPTIMIZATION (from GRPO blog)
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
    \
    # LOGGING & MONITORING (WandB integration)
    trainer.logger='["console","wandb"]' \
    trainer.project_name='medserl-selfplay' \
    trainer.experiment_name='injector-assessor-game' \
    trainer.log_freq=5 \
    trainer.test_freq=5 \
    \
    # CHECKPOINTING (recovery capability)
    trainer.total_epochs=$RL_EPISODES \
    trainer.save_freq=5 \
    trainer.save_total_limit=3 \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.resume_mode="auto" \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    \
    # ALGORITHM CONFIGURATION
    algorithm.adv_estimator=gae \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    # LEARNING RATES
    critic.optim.lr=$LEARNING_RATE \
    actor.optim.lr=$LEARNING_RATE

# ============================================================================
# TRAINING COMPLETE
# ============================================================================

echo ""
echo "=================================================="
echo "✅ Training complete!"
echo ""
echo "📁 Outputs saved to: $OUTPUT_DIR"
echo "📊 WandB project: https://wandb.ai/home"
echo ""
echo "📈 Next steps:"
echo "   1. Check WandB dashboard for experiments"
echo "   2. Compare metrics with baseline"
echo "   3. Run inference on test set"
echo "=================================================="
