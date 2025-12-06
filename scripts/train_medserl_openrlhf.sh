#!/bin/bash
# =============================================================================
# MedSeRL Training Script with OpenRLHF Integration
# =============================================================================
#
# This script configures and runs MedSeRL training using the OpenRLHF framework
# with Reinforce++ algorithm for policy optimization.
#
# Adapted from SeRL template for medical error detection on MEDEC dataset.
#
# Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
#
# Usage:
#   ./scripts/train_medserl_openrlhf.sh --model_path /path/to/medgemma
#
# For multi-GPU setup:
#   ./scripts/train_medserl_openrlhf.sh --model_path /path/to/medgemma --num_gpus 4
#
# =============================================================================

set -e
set -x

# Start time tracking
start_time=$(date +%s)

# =============================================================================
# Default Configuration
# =============================================================================

# Model paths
MODEL_PATH="${MODEL_PATH:-}"
SCRIBE_MODEL_PATH="${SCRIBE_MODEL_PATH:-}"  # Defaults to MODEL_PATH if not set

# Data paths
MEDEC_PATH="${MEDEC_PATH:-data_raw/MEDEC}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/medserl}"

# Reward server configuration (Requirements: 9.4)
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-8000}"
REWARD_SERVER_HOST="${REWARD_SERVER_HOST:-localhost}"

# =============================================================================
# Ray Configuration (Requirements: 9.1)
# =============================================================================
RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-8}"

# =============================================================================
# Resource Allocations (Requirements: 9.2)
# =============================================================================
# Actor model (policy being trained)
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS="${ACTOR_NUM_GPUS:-1}"

# Reference model (for KL divergence)
REF_NUM_NODES="${REF_NUM_NODES:-1}"
REF_NUM_GPUS="${REF_NUM_GPUS:-1}"

# Reward model (uses our custom reward server)
REWARD_NUM_NODES="${REWARD_NUM_NODES:-1}"
REWARD_NUM_GPUS="${REWARD_NUM_GPUS:-0}"  # CPU-based reward calculation

# Critic model (for value estimation)
CRITIC_NUM_NODES="${CRITIC_NUM_NODES:-1}"
CRITIC_NUM_GPUS="${CRITIC_NUM_GPUS:-1}"

# =============================================================================
# vLLM Configuration (Requirements: 9.3)
# =============================================================================
VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-1}"
VLLM_TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL:-1}"
VLLM_GPU_MEMORY="${VLLM_GPU_MEMORY:-0.85}"
VLLM_SYNC_BACKEND="${VLLM_SYNC_BACKEND:-nccl}"

# =============================================================================
# Training Hyperparameters
# =============================================================================
NUM_EPISODES="${NUM_EPISODES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-4}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"

# Learning rates
ACTOR_LEARNING_RATE="${ACTOR_LEARNING_RATE:-5e-7}"
CRITIC_LEARNING_RATE="${CRITIC_LEARNING_RATE:-9e-6}"

# KL divergence control
KL_COEF="${KL_COEF:-1e-4}"

# Sequence lengths
PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-1024}"
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-1024}"

# Generation parameters
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"

# =============================================================================
# Schedule Configuration
# =============================================================================
EVAL_STEPS="${EVAL_STEPS:-50}"
SAVE_STEPS="${SAVE_STEPS:-100}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-100000}"

# =============================================================================
# Logging Configuration
# =============================================================================
USE_WANDB="${USE_WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-medserl}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-medserl_$(date +%Y%m%d_%H%M%S)}"

# =============================================================================
# Advanced Options
# =============================================================================
ZERO_STAGE="${ZERO_STAGE:-3}"
COLOCATE_MODELS="${COLOCATE_MODELS:-true}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-true}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
ADAM_OFFLOAD="${ADAM_OFFLOAD:-true}"
PACKING_SAMPLES="${PACKING_SAMPLES:-true}"
NORMALIZE_REWARD="${NORMALIZE_REWARD:-true}"
ENFORCE_EAGER="${ENFORCE_EAGER:-true}"
VLLM_ENABLE_SLEEP="${VLLM_ENABLE_SLEEP:-true}"
DEEPSPEED_ENABLE_SLEEP="${DEEPSPEED_ENABLE_SLEEP:-true}"

# =============================================================================
# Parse Command Line Arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --scribe_model_path)
            SCRIBE_MODEL_PATH="$2"
            shift 2
            ;;
        --medec_path)
            MEDEC_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --rollout_batch_size)
            ROLLOUT_BATCH_SIZE="$2"
            shift 2
            ;;
        --actor_learning_rate)
            ACTOR_LEARNING_RATE="$2"
            shift 2
            ;;
        --kl_coef)
            KL_COEF="$2"
            shift 2
            ;;
        --num_gpus)
            # Set all GPU counts to this value
            ACTOR_NUM_GPUS="$2"
            REF_NUM_GPUS="$2"
            CRITIC_NUM_GPUS="$2"
            VLLM_NUM_ENGINES="$2"
            shift 2
            ;;
        --vllm_num_engines)
            VLLM_NUM_ENGINES="$2"
            shift 2
            ;;
        --vllm_tensor_parallel)
            VLLM_TENSOR_PARALLEL="$2"
            shift 2
            ;;
        --ray_address)
            RAY_ADDRESS="$2"
            shift 2
            ;;
        --reward_server_port)
            REWARD_SERVER_PORT="$2"
            shift 2
            ;;
        --use_wandb)
            USE_WANDB="true"
            shift
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb_run_name)
            WANDB_RUN_NAME="$2"
            shift 2
            ;;
        --eval_steps)
            EVAL_STEPS="$2"
            shift 2
            ;;
        --save_steps)
            SAVE_STEPS="$2"
            shift 2
            ;;
        --no_colocate)
            COLOCATE_MODELS="false"
            shift
            ;;
        --help)
            echo "MedSeRL Training Script with OpenRLHF"
            echo ""
            echo "Usage: $0 --model_path <path> [options]"
            echo ""
            echo "Required:"
            echo "  --model_path           Path to base model (MedGemma-4B)"
            echo ""
            echo "Data Paths:"
            echo "  --medec_path           Path to MEDEC dataset (default: data_raw/MEDEC)"
            echo "  --output_dir           Output directory (default: outputs/medserl)"
            echo ""
            echo "Training Configuration:"
            echo "  --num_episodes         Number of training episodes (default: 1000)"
            echo "  --batch_size           Training batch size (default: 16)"
            echo "  --rollout_batch_size   Rollout batch size (default: 256)"
            echo "  --actor_learning_rate  Actor learning rate (default: 5e-7)"
            echo "  --kl_coef              KL coefficient (default: 1e-4)"
            echo ""
            echo "Infrastructure:"
            echo "  --num_gpus             Number of GPUs per component (default: 1)"
            echo "  --vllm_num_engines     Number of vLLM engines (default: 1)"
            echo "  --vllm_tensor_parallel Tensor parallel size (default: 1)"
            echo "  --ray_address          Ray cluster address (default: http://127.0.0.1:8265)"
            echo "  --reward_server_port   Reward server port (default: 8000)"
            echo "  --no_colocate          Disable model colocation"
            echo ""
            echo "Logging:"
            echo "  --use_wandb            Enable Weights & Biases logging"
            echo "  --wandb_project        W&B project name (default: medserl)"
            echo "  --wandb_run_name       W&B run name"
            echo ""
            echo "Schedule:"
            echo "  --eval_steps           Evaluate every N steps (default: 50)"
            echo "  --save_steps           Save checkpoint every N steps (default: 100)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# Validate Required Arguments
# =============================================================================
if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model_path is required"
    echo "Use --help for usage information"
    exit 1
fi

# Default scribe model to main model if not specified
if [ -z "$SCRIBE_MODEL_PATH" ]; then
    SCRIBE_MODEL_PATH="$MODEL_PATH"
fi

# Validate paths exist
if [ ! -d "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

if [ ! -d "$MEDEC_PATH" ]; then
    echo "Error: MEDEC data path does not exist: $MEDEC_PATH"
    exit 1
fi

# =============================================================================
# Create Output Directories
# =============================================================================
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/eval"
mkdir -p "$OUTPUT_DIR/train_samples"
mkdir -p "$OUTPUT_DIR/filtered_data"

# =============================================================================
# Print Configuration
# =============================================================================
echo "============================================================"
echo "MedSeRL Training with OpenRLHF (Reinforce++)"
echo "============================================================"
echo ""
echo "Model Configuration:"
echo "  Base Model:        $MODEL_PATH"
echo "  Scribe Model:      $SCRIBE_MODEL_PATH"
echo ""
echo "Data Configuration:"
echo "  MEDEC Path:        $MEDEC_PATH"
echo "  Output Dir:        $OUTPUT_DIR"
echo ""
echo "Training Configuration:"
echo "  Num Episodes:      $NUM_EPISODES"
echo "  Batch Size:        $BATCH_SIZE"
echo "  Rollout Batch:     $ROLLOUT_BATCH_SIZE"
echo "  Actor LR:          $ACTOR_LEARNING_RATE"
echo "  KL Coefficient:    $KL_COEF"
echo ""
echo "Infrastructure:"
echo "  Actor GPUs:        $ACTOR_NUM_GPUS"
echo "  vLLM Engines:      $VLLM_NUM_ENGINES"
echo "  Tensor Parallel:   $VLLM_TENSOR_PARALLEL"
echo "  Ray Address:       $RAY_ADDRESS"
echo ""
echo "Schedule:"
echo "  Eval Steps:        $EVAL_STEPS"
echo "  Save Steps:        $SAVE_STEPS"
echo "============================================================"
echo ""

# =============================================================================
# Start Reward Server (Requirements: 9.4)
# =============================================================================
echo "Starting MedSeRL reward server on port $REWARD_SERVER_PORT..."

# Kill any existing reward server on this port
pkill -f "reward_server.*--port $REWARD_SERVER_PORT" 2>/dev/null || true
sleep 1

# Start reward server in background
python -m src.training.reward_server \
    --host "0.0.0.0" \
    --port "$REWARD_SERVER_PORT" \
    --medec_path "$MEDEC_PATH" \
    > "$OUTPUT_DIR/logs/reward_server.log" 2>&1 &
REWARD_SERVER_PID=$!

# Wait for server to start
echo "Waiting for reward server to start..."
sleep 5

# Check if reward server is running
if ! kill -0 $REWARD_SERVER_PID 2>/dev/null; then
    echo "Error: Reward server failed to start"
    echo "Check logs at: $OUTPUT_DIR/logs/reward_server.log"
    cat "$OUTPUT_DIR/logs/reward_server.log"
    exit 1
fi

# Verify server is responding
if command -v curl &> /dev/null; then
    if ! curl -s "http://${REWARD_SERVER_HOST}:${REWARD_SERVER_PORT}/health" > /dev/null; then
        echo "Warning: Reward server health check failed, but process is running"
    else
        echo "Reward server is healthy"
    fi
fi

echo "Reward server started (PID: $REWARD_SERVER_PID)"

# =============================================================================
# Cleanup Function
# =============================================================================
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "$REWARD_SERVER_PID" ]; then
        echo "Stopping reward server (PID: $REWARD_SERVER_PID)..."
        kill $REWARD_SERVER_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT

# =============================================================================
# Build OpenRLHF Command (Requirements: 9.5)
# =============================================================================

# Base command arguments
CMD_ARGS=(
    # Model configuration
    --pretrain "$MODEL_PATH"
    --save_path "$OUTPUT_DIR/checkpoints"
    --ckpt_path "$OUTPUT_DIR/checkpoints"
    --save_hf_ckpt
    
    # Reward configuration (Requirements: 9.4)
    --remote_rm_url "http://${REWARD_SERVER_HOST}:${REWARD_SERVER_PORT}/reward"
    
    # Resource allocation (Requirements: 9.2)
    --actor_num_nodes "$ACTOR_NUM_NODES"
    --actor_num_gpus_per_node "$ACTOR_NUM_GPUS"
    --ref_num_nodes "$REF_NUM_NODES"
    --ref_num_gpus_per_node "$REF_NUM_GPUS"
    --reward_num_nodes "$REWARD_NUM_NODES"
    --reward_num_gpus_per_node "$REWARD_NUM_GPUS"
    
    # vLLM configuration (Requirements: 9.3)
    --vllm_num_engines "$VLLM_NUM_ENGINES"
    --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL"
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY"
    --vllm_sync_backend "$VLLM_SYNC_BACKEND"
    
    # Algorithm configuration (Requirements: 9.5)
    --advantage_estimator "reinforce"
    --init_kl_coef "$KL_COEF"
    
    # Training hyperparameters
    --actor_learning_rate "$ACTOR_LEARNING_RATE"
    --critic_learning_rate "$CRITIC_LEARNING_RATE"
    --micro_train_batch_size "$MICRO_BATCH_SIZE"
    --train_batch_size "$BATCH_SIZE"
    --micro_rollout_batch_size "$MICRO_ROLLOUT_BATCH_SIZE"
    --rollout_batch_size "$ROLLOUT_BATCH_SIZE"
    --n_samples_per_prompt "$N_SAMPLES_PER_PROMPT"
    
    # Sequence configuration
    --prompt_max_len "$PROMPT_MAX_LEN"
    --generate_max_len "$GENERATE_MAX_LEN"
    
    # Schedule
    --max_epochs "$MAX_EPOCHS"
    --max_samples "$MAX_SAMPLES"
    --num_episodes "$NUM_EPISODES"
    --eval_steps "$EVAL_STEPS"
    --save_steps "$SAVE_STEPS"
    --logging_steps "$LOGGING_STEPS"
    
    # DeepSpeed configuration
    --zero_stage "$ZERO_STAGE"
    
    # Precision
    --bf16
    
    # Output directories
    --eval_output_root_dir "$OUTPUT_DIR/eval"
    --train_samples_root_dir "$OUTPUT_DIR/train_samples"
    --filtered_data_root_dir "$OUTPUT_DIR/filtered_data"
)

# Add optional flags
if [ "$COLOCATE_MODELS" = "true" ]; then
    CMD_ARGS+=(--colocate_all_models)
fi

if [ "$USE_FLASH_ATTN" = "true" ]; then
    CMD_ARGS+=(--flash_attn)
fi

if [ "$GRADIENT_CHECKPOINTING" = "true" ]; then
    CMD_ARGS+=(--gradient_checkpointing)
fi

if [ "$ADAM_OFFLOAD" = "true" ]; then
    CMD_ARGS+=(--adam_offload)
fi

if [ "$PACKING_SAMPLES" = "true" ]; then
    CMD_ARGS+=(--packing_samples)
fi

if [ "$NORMALIZE_REWARD" = "true" ]; then
    CMD_ARGS+=(--normalize_reward)
fi

if [ "$ENFORCE_EAGER" = "true" ]; then
    CMD_ARGS+=(--enforce_eager)
fi

if [ "$VLLM_ENABLE_SLEEP" = "true" ]; then
    CMD_ARGS+=(--vllm_enable_sleep)
fi

if [ "$DEEPSPEED_ENABLE_SLEEP" = "true" ]; then
    CMD_ARGS+=(--deepspeed_enable_sleep)
fi

# Add W&B configuration
if [ "$USE_WANDB" = "true" ]; then
    CMD_ARGS+=(
        --use_wandb "$WANDB_PROJECT"
        --wandb_project "$WANDB_PROJECT"
        --wandb_run_name "$WANDB_RUN_NAME"
    )
fi

# =============================================================================
# Run Training (Requirements: 9.1, 9.5)
# =============================================================================
echo ""
echo "Starting OpenRLHF training with Reinforce++..."
echo ""

# Get the workspace directory (parent of scripts/)
WORKSPACE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Submit job to Ray cluster (Requirements: 9.1)
ray job submit --address="$RAY_ADDRESS" \
    --runtime-env-json="{\"working_dir\": \"$WORKSPACE_DIR\", \"excludes\": [\"outputs/\", \".git/\", \"__pycache__/\", \"*.pyc\", \".hypothesis/\"]}" \
    -- python3 -m openrlhf.cli.train_ppo_ray "${CMD_ARGS[@]}"

# =============================================================================
# Training Complete
# =============================================================================
end_time=$(date +%s)
execution_time=$((end_time - start_time))

echo ""
echo "============================================================"
echo "Training Complete!"
echo "============================================================"
echo "Execution time: $execution_time seconds"
echo "Checkpoints saved to: $OUTPUT_DIR/checkpoints"
echo "Logs saved to: $OUTPUT_DIR/logs"
echo "============================================================"
