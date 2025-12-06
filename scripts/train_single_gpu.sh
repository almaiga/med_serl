#!/bin/bash
# =============================================================================
# MedSeRL Single GPU Training Script
# =============================================================================
#
# Convenience script for training MedSeRL on a single GPU setup.
# Uses the single_gpu.yaml configuration file.
#
# Requirements: 10.1, 10.2, 10.3
#
# Usage:
#   ./scripts/train_single_gpu.sh --model_path /path/to/medgemma
#
# =============================================================================

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
MODEL_PATH="${MODEL_PATH:-}"
MEDEC_PATH="${MEDEC_PATH:-data_raw/MEDEC}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/medserl_single_gpu}"

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
        --help)
            echo "MedSeRL Single GPU Training"
            echo ""
            echo "Usage: $0 --model_path <path> [options]"
            echo ""
            echo "Options:"
            echo "  --model_path    Path to MedGemma-4B model (required)"
            echo "  --medec_path    Path to MEDEC dataset (default: data_raw/MEDEC)"
            echo "  --output_dir    Output directory (default: outputs/medserl_single_gpu)"
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
export NUM_GPUS=1

echo "============================================================"
echo "MedSeRL Single GPU Training"
echo "============================================================"
echo "Model Path:  $MODEL_PATH"
echo "MEDEC Path:  $MEDEC_PATH"
echo "Output Dir:  $OUTPUT_DIR"
echo "============================================================"

# Run training with single GPU configuration
"$SCRIPT_DIR/train_medserl_openrlhf.sh" \
    --model_path "$MODEL_PATH" \
    --medec_path "$MEDEC_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_gpus 1 \
    --batch_size 4 \
    --rollout_batch_size 16 \
    --vllm_num_engines 1
