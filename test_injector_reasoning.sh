#!/bin/bash
# Test injector reasoning generation

echo "Testing injector reasoning with 2 pairs..."

python scripts/generate_sft_data_safe.py \
    --input-jsonl data_processed/medec_paired/train_val_split/sft_train.jsonl \
    --output-dir data/sft_injector_reasoning_test \
    --prompt-file configs/prompts/gpt4o_medec_safe_augmentation.json \
    --num-pairs 2 \
    --add-reasoning \
    --openai-api-key "$OPENAI_API_KEY"

echo ""
echo "=========================================="
echo "Sample INCORRECT with Injector Reasoning:"
echo "=========================================="
head -n 1 data/sft_injector_reasoning_test/sft_incorrect.jsonl | jq '.reasoning'
