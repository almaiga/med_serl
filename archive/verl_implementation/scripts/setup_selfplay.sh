#!/bin/bash
# Quick Setup Script for MedSeRL Self-Play Training
# Run this to set up everything from scratch

set -e

PROJECT_ROOT="/Users/josmaiga/Documents/GitHub/med_serl"
cd "$PROJECT_ROOT"

echo "=================================================="
echo "MedSeRL Self-Play Setup"
echo "=================================================="

# Step 1: Check data exists
echo -e "\n[1/4] Checking input data..."
if [ ! -f "data_processed/medec_paired/train_val_split/rl_train.jsonl" ]; then
    echo "❌ Input file not found: rl_train.jsonl"
    exit 1
fi
echo "✓ Input data found (405 pairs)"

# Step 2: Generate self-play data
echo -e "\n[2/4] Generating self-play data (810 examples)..."
python3 verl_implementation/data/preprocess_selfplay.py \
    --input data_processed/medec_paired/train_val_split/rl_train.jsonl \
    --output_dir data_processed/selfplay

echo "✓ Generated 810 examples (405 benign + 405 error)"

# Step 3: Verify data
echo -e "\n[3/4] Verifying parquet files..."
python3 verl_implementation/scripts/verify_data.py

# Step 4: Test interaction
echo -e "\n[4/4] Testing interaction system..."
python3 verl_implementation/scripts/test_interaction.py

echo ""
echo "=================================================="
echo "✓ Setup Complete!"
echo "=================================================="
echo ""
echo "To start training, run:"
echo "  bash verl_implementation/scripts/run_training.sh"
echo ""
echo "To monitor training:"
echo "  - Check console output"
echo "  - View W&B: https://wandb.ai/[your-entity]/medserl-selfplay"
echo ""
