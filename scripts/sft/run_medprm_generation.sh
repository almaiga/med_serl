#!/bin/bash
# =============================================================================
# Run Med-PRM Chain Generation with QwenMax
# =============================================================================
#
# Usage:
#   ./scripts/sft/run_medprm_generation.sh           # Process all pairs
#   ./scripts/sft/run_medprm_generation.sh 10        # Test with 10 pairs
#   ./scripts/sft/run_medprm_generation.sh 0 100     # Resume from index 100
#
# =============================================================================

set -e

# Configuration
INPUT="data_processed/medec_paired/train_val_split/sft_train.jsonl"
PROMPTS="configs/prompts/sft/medprm_prompts.json"
OUTPUT="data_processed/medprm_chains"
MODEL="qwen-max"
CONCURRENCY=5
CHECKPOINT_EVERY=25

# Parse arguments
LIMIT=${1:-}
RESUME=${2:-0}

echo "=========================================="
echo "  Med-PRM Chain Generation with QwenMax"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Input:        $INPUT"
echo "  Prompts:      $PROMPTS"
echo "  Output:       $OUTPUT"
echo "  Model:        $MODEL"
echo "  Concurrency:  $CONCURRENCY"
echo "  Checkpoint:   Every $CHECKPOINT_EVERY pairs"
echo ""
echo "Parameters:"
echo "  Limit:        ${LIMIT:-'all pairs'}"
echo "  Resume from:  $RESUME"
echo ""

# Check files exist
if [ ! -f "$INPUT" ]; then
    echo "❌ Error: Input file not found: $INPUT"
    exit 1
fi

if [ ! -f "$PROMPTS" ]; then
    echo "❌ Error: Prompts file not found: $PROMPTS"
    exit 1
fi

# Check API key
if [ -z "$QWEN_API_KEY" ]; then
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs)
    fi
fi

if [ -z "$QWEN_API_KEY" ]; then
    echo "❌ Error: QWEN_API_KEY not set"
    echo "   Set it in .env file or export QWEN_API_KEY=..."
    exit 1
fi

echo "✓ QWEN_API_KEY found"
echo ""

# Create output directory
mkdir -p "$OUTPUT"

# Count pairs
PAIR_COUNT=$(wc -l < "$INPUT" | tr -d ' ')
echo "📊 Dataset: $PAIR_COUNT pairs"
echo "📝 Expected chains: ~$((PAIR_COUNT * 3))"
echo ""

# Build command
CMD="python scripts/sft/generate_medprm_chains.py"
CMD="$CMD --input $INPUT"
CMD="$CMD --prompts $PROMPTS"
CMD="$CMD --output $OUTPUT"
CMD="$CMD --model $MODEL"
CMD="$CMD --concurrency $CONCURRENCY"
CMD="$CMD --checkpoint-every $CHECKPOINT_EVERY"

if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

if [ "$RESUME" -gt 0 ]; then
    CMD="$CMD --resume $RESUME"
fi

echo "🚀 Starting generation..."
echo "   Command: $CMD"
echo ""
echo "=========================================="

# Run
$CMD

echo ""
echo "=========================================="
echo "✅ Generation complete!"
echo "   Output directory: $OUTPUT"
echo "=========================================="
