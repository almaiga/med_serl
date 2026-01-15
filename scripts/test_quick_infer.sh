#!/bin/bash
#
# Test quick_infer_qwen3_4b_lora.py with your trained model
# Uses your production prompts and proper Qwen thinking mode
#

set -e

echo "============================================================"
echo "Testing quick_infer_qwen3_4b_lora.py on M3 Max"
echo "============================================================"

# Configuration
MODEL_NAME="google/medgemma-4b-it"  # Your adapter base model
ADAPTER_DIR="/Users/josmaiga/Documents/GitHub/med_serl/outputs/local_training/sft/sft_checkpoint"
PROMPT_DIR="/Users/josmaiga/Documents/GitHub/med_serl/configs/prompts"

# Test with a few samples
JSONL_FILE="/Users/josmaiga/Documents/GitHub/med_serl/data_processed/medec_paired/train_val_split/rl_train.jsonl"
NUM_SAMPLES=2

OUTPUT_DIR="outputs/test_quick_infer"

echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Adapter: $ADAPTER_DIR"
echo "  Prompts: $PROMPT_DIR"
echo "  Samples: $NUM_SAMPLES"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run quick_infer with your settings
echo "Running quick_infer_qwen3_4b_lora.py..."
echo ""

python scripts/sft/quick_infer_qwen3_4b_lora.py \
    --model-name "$MODEL_NAME" \
    --adapter-dir "$ADAPTER_DIR" \
    --jsonl-file "$JSONL_FILE" \
    --num-samples "$NUM_SAMPLES" \
    --scenarios "injector_incorrect,assessor_correct" \
    --assessor-prompt-file "$PROMPT_DIR/error_detection_prompts.json" \
    --injector-prompt-file "$PROMPT_DIR/error_injection_prompts_v2.json" \
    --output-dir "$OUTPUT_DIR" \
    --thinking-budget 512 \
    --assessor-thinking-budget 256 \
    --max-new-tokens 1024 \
    --batch-size 1

echo ""
echo "============================================================"
echo "✅ Test complete!"
echo "============================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Check outputs:"
ls -lh "$OUTPUT_DIR"/*.jsonl 2>/dev/null || echo "  (no output files yet)"
