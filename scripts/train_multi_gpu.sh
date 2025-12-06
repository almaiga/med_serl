#!/bin/bash
# =============================================================================
# MedSeRL Multi-GPU Training Script
# =============================================================================
#
# Convenience script for training MedSeRL on a multi-GPU setup.
# Uses the multi_gpu.yaml configuration file.
#
# Requirements: 10.1, 10.2, 10.3
#
# Usage:
#   ./scripts/train_multi_gpu.sh --model_path /path/to/medgemma --num_gpus 4
#
# =============================================================================

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
MODEL_PATH="${MODEL_PATH:-}"
MEDEC_PATH="${MEDEC_PATH:-data_raw/MEDEC}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/medserl_multi_gpu}"
NUM_GPUS="${NUM_GPUS:-4}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
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
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --use_wandb)
            USE_WANDB="true"
            shift
            ;;
        --help)
            echo "MedSeRL Multi-GPU Training"
            echo ""
            echo "Usage: $0 --model_path <path> [options]"
            echo ""
            echo "Options:"
            echo "  --model_path    Path to MedGemma-4B model (required)"
            echo "  --medec_path    Path to MEDEC dataset (default: data_raw/MEDEC)"
            echo "  --output_dir    Output directory (default: outputs/medserl_multi_gpu)"
            echo "  --num_gpus      Number of GPUs to use (default: 4)"
            echo "  --use_wandb     Enable Weights & Biases logging"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model_path is required"
    exit 1
fi

# Export environment variables for config loader
export MODEL_PATH
export MEDEC_PATH
export OUTPUT_DIR
export NUM_GPUS

echo "============================================================"
echo "MedSeRL Multi-GPU Training"
echo "============================================================"
echo "Model Path:  $MODEL_PATH"
echo "MEDEC Path:  $MEDEC_PATH"
echo "Output Dir:  $OUTPUT_DIR"
echo "Num GPUs:    $NUM_GPUS"
echo "============================================================"

# Calculate batch sizes based on GPU count
BATCH_SIZE=$((NUM_GPUS * 4))  # 4 samples per GPU (must be divisible by 4)
ROLLOUT_BATCH_SIZE=$((NUM_GPUS * 16))  # 16 rollouts per GPU

# Build command
CMD_ARGS=(
    --model_path "$MODEL_PATH"
    --medec_path "$MEDEC_PATH"
    --output_dir "$OUTPUT_DIR"
    --num_gpus "$NUM_GPUS"
    --batch_size "$BATCH_SIZE"
    --rollout_batch_size "$ROLLOUT_BATCH_SIZE"
    --vllm_num_engines "$NUM_GPUS"
)

# Add W&B if requested
if [ "$USE_WANDB" = "true" ]; then
    CMD_ARGS+=(--use_wandb)
fi

# Run training with multi-GPU configuration
"$SCRIPT_DIR/train_medserl_openrlhf.sh" "${CMD_ARGS[@]}"
