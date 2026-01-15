# Batch Processing & Checkpoint System

## New Features ✨

### 1. **Incremental Checkpoint Saves**
- Saves progress every N pairs (default: 5)
- Prevents data loss if API fails or script crashes
- Auto-resume from last checkpoint

### 2. **Full Path Resolution**
- All paths converted to absolute paths
- Can run script from anywhere on your computer
- No more "file not found" errors

### 3. **Error Recovery**
- Catches API errors and continues
- Saves checkpoint on error
- Detailed error logging

### 4. **Batch Processing**
- Configurable batch size (default: 10)
- Memory-efficient processing
- Progress tracking with stats

---

## Usage

### Basic Command (with full paths shown)

```bash
python3 /full/path/to/scripts/generate_sft_data_safe.py \
    --input-jsonl /full/path/to/data_processed/medec_paired/train_val_split/sft_train.jsonl \
    --output-dir /full/path/to/data/sft_production_v1 \
    --prompt-file /full/path/to/configs/prompts/gpt4o_medec_safe_augmentation.json \
    --num-pairs 500 \
    --add-reasoning \
    --save-every 5 \
    --openai-api-key "$OPENAI_API_KEY"
```

### Run from Anywhere (using relative paths - will be converted to absolute)

```bash
# From project root
cd /Users/josmaiga/.claude-worktrees/sft/pedantic-hertz

python3 scripts/generate_sft_data_safe.py \
    --input-jsonl data_processed/medec_paired/train_val_split/sft_train.jsonl \
    --output-dir data/sft_production_v1 \
    --prompt-file configs/prompts/gpt4o_medec_safe_augmentation.json \
    --num-pairs 500 \
    --add-reasoning \
    --save-every 5 \
    --openai-api-key "$OPENAI_API_KEY"
```

### Run from Different Directory

```bash
# From anywhere on your computer
cd ~/Documents

python3 /Users/josmaiga/.claude-worktrees/sft/pedantic-hertz/scripts/generate_sft_data_safe.py \
    --input-jsonl /Users/josmaiga/.claude-worktrees/sft/pedantic-hertz/data_processed/medec_paired/train_val_split/sft_train.jsonl \
    --output-dir /Users/josmaiga/.claude-worktrees/sft/pedantic-hertz/data/sft_production_v1 \
    --prompt-file /Users/josmaiga/.claude-worktrees/sft/pedantic-hertz/configs/prompts/gpt4o_medec_safe_augmentation.json \
    --num-pairs 500 \
    --add-reasoning \
    --save-every 5 \
    --openai-api-key "$OPENAI_API_KEY"
```

---

## Command-Line Arguments

### Required:
- `--input-jsonl`: Path to MEDEC paired JSONL file
- `--output-dir`: Output directory for generated data
- `--prompt-file`: Path to prompt configuration JSON

### Optional:
- `--num-pairs`: Number of pairs to process (default: all)
- `--add-reasoning`: Add chain-of-thought reasoning (recommended)
- `--batch-size`: Batch size for processing (default: 10)
- `--save-every`: Save checkpoint every N pairs (default: 5)
- `--openai-api-key`: OpenAI API key (or use env var)

---

## Checkpoint System

### How It Works

1. **Every N pairs (default: 5):**
   - Flushes file buffers (saves data to disk)
   - Creates checkpoint file: `output_dir/checkpoint.json`
   - Prints progress message

2. **Checkpoint file contains:**
   ```json
   {
     "last_completed_idx": 49,
     "stats": {
       "correct_generated": 50,
       "incorrect_used": 50,
       "total_tokens_used": 95000
     },
     "timestamp": "2026-01-15 14:30:45"
   }
   ```

3. **On script interruption:**
   - Data up to last checkpoint is saved
   - Next run auto-resumes from checkpoint
   - No data loss!

### Resume from Checkpoint

If script crashes or is interrupted, simply re-run the same command:

```bash
# Same command as before
python3 scripts/generate_sft_data_safe.py \
    --input-jsonl data_processed/medec_paired/train_val_split/sft_train.jsonl \
    --output-dir data/sft_production_v1 \
    --prompt-file configs/prompts/gpt4o_medec_safe_augmentation.json \
    --num-pairs 500 \
    --add-reasoning \
    --save-every 5 \
    --openai-api-key "$OPENAI_API_KEY"

# Script detects checkpoint and prints:
# "Resuming from pair 50 (checkpoint found)"
```

---

## Error Handling

### API Errors

If OpenAI API fails:
```
❌ Error processing pair 123 (ms-train-456): API rate limit exceeded
Saving checkpoint and continuing...
✓ Checkpoint saved at pair 122/500
```

Script will:
1. Save checkpoint at last successful pair
2. Continue with next pair
3. Log error details

### Recovery

After error, you can:
1. **Wait and retry** - API rate limits usually reset quickly
2. **Resume from checkpoint** - Re-run same command
3. **Check logs** - Review error messages

---

## Output Files

### During Generation

Files are updated incrementally every `--save-every` pairs:

```
data/sft_production_v1/
├── sft_correct.jsonl          # CORRECT examples (incremental)
├── sft_incorrect.jsonl        # INCORRECT examples (incremental)
├── sft_combined_safe.jsonl    # Combined 50/50 (incremental)
└── checkpoint.json            # Current progress (updated every N pairs)
```

### After Completion

```
data/sft_production_v1/
├── sft_correct.jsonl          # 500 CORRECT examples (final)
├── sft_incorrect.jsonl        # 500 INCORRECT examples (final)
├── sft_combined_safe.jsonl    # 1000 total examples (final)
└── generation_stats_safe.json # Final statistics
```

**Note:** `checkpoint.json` is deleted on successful completion.

---

## Full Path Examples

### Recommended: Set as Environment Variable

```bash
# Add to ~/.bashrc or ~/.zshrc
export SFT_PROJECT_ROOT="/Users/josmaiga/.claude-worktrees/sft/pedantic-hertz"

# Then use:
python3 $SFT_PROJECT_ROOT/scripts/generate_sft_data_safe.py \
    --input-jsonl $SFT_PROJECT_ROOT/data_processed/medec_paired/train_val_split/sft_train.jsonl \
    --output-dir $SFT_PROJECT_ROOT/data/sft_production_v1 \
    --prompt-file $SFT_PROJECT_ROOT/configs/prompts/gpt4o_medec_safe_augmentation.json \
    --num-pairs 500 \
    --add-reasoning \
    --save-every 5 \
    --openai-api-key "$OPENAI_API_KEY"
```

### Alternative: Create Shell Script

```bash
# Create run_sft_generation.sh
cat > /Users/josmaiga/.claude-worktrees/sft/pedantic-hertz/run_sft_generation.sh << 'EOF'
#!/bin/bash

PROJECT_ROOT="/Users/josmaiga/.claude-worktrees/sft/pedantic-hertz"

python3 "$PROJECT_ROOT/scripts/generate_sft_data_safe.py" \
    --input-jsonl "$PROJECT_ROOT/data_processed/medec_paired/train_val_split/sft_train.jsonl" \
    --output-dir "$PROJECT_ROOT/data/sft_production_v1" \
    --prompt-file "$PROJECT_ROOT/configs/prompts/gpt4o_medec_safe_augmentation.json" \
    --num-pairs 500 \
    --add-reasoning \
    --save-every 5 \
    --openai-api-key "$OPENAI_API_KEY"
EOF

chmod +x /Users/josmaiga/.claude-worktrees/sft/pedantic-hertz/run_sft_generation.sh

# Run from anywhere:
/Users/josmaiga/.claude-worktrees/sft/pedantic-hertz/run_sft_generation.sh
```

---

## Progress Monitoring

### Console Output

```
Input file: /Users/josmaiga/.claude-worktrees/sft/pedantic-hertz/data_processed/medec_paired/train_val_split/sft_train.jsonl
Output directory: /Users/josmaiga/.claude-worktrees/sft/pedantic-hertz/data/sft_production_v1
Prompt file: /Users/josmaiga/.claude-worktrees/sft/pedantic-hertz/configs/prompts/gpt4o_medec_safe_augmentation.json

Loading MEDEC pairs...
Loaded 500 MEDEC pairs from ...

Generating safe SFT data...
Strategy: Use MEDEC errors AS-IS, generate CORRECT paraphrases
Output: 50/50 CORRECT/INCORRECT split
Adding clinical reasoning to INCORRECT examples

Generating SFT data:   2%|██                | 10/500 [00:42<35:20, 4.33s/it]
✓ Checkpoint saved at pair 10/500

Generating SFT data:   4%|████              | 20/500 [01:24<34:38, 4.33s/it]
✓ Checkpoint saved at pair 20/500
```

### Check Progress During Run

```bash
# Count completed examples
wc -l data/sft_production_v1/sft_correct.jsonl
wc -l data/sft_production_v1/sft_incorrect.jsonl

# View latest checkpoint
cat data/sft_production_v1/checkpoint.json | jq '.'

# Tail recent examples
tail -1 data/sft_production_v1/sft_combined_safe.jsonl | jq '.'
```

---

## Best Practices

### 1. Start Small
```bash
# Test with 2 pairs first
python3 scripts/generate_sft_data_safe.py \
    --num-pairs 2 \
    --save-every 1 \
    # ... other args
```

### 2. Use Frequent Checkpoints for Large Runs
```bash
# For 500 pairs, save every 5-10 pairs
python3 scripts/generate_sft_data_safe.py \
    --num-pairs 500 \
    --save-every 5 \
    # ... other args
```

### 3. Monitor Token Usage
```bash
# After each checkpoint, check stats
cat data/sft_production_v1/checkpoint.json | jq '.stats.total_tokens_used'

# Estimate cost (GPT-4o: ~$0.015 per 1k tokens)
# Example: 100k tokens = ~$1.50
```

### 4. Set Environment Variable for API Key
```bash
# More secure than command-line argument
export OPENAI_API_KEY="sk-..."

python3 scripts/generate_sft_data_safe.py \
    # ... other args (no --openai-api-key needed)
```

---

## Troubleshooting

### Issue: "File not found"
**Solution:** Use absolute paths or run from project root
```bash
# Check current directory
pwd

# Use absolute paths
python3 /full/path/to/script.py --input-jsonl /full/path/to/input.jsonl
```

### Issue: API rate limit
**Solution:** Script auto-saves checkpoint and continues
```bash
# Wait a bit, then re-run same command
# Script resumes from last checkpoint
```

### Issue: Checkpoint not resuming
**Solution:** Check checkpoint file exists
```bash
ls -la data/sft_production_v1/checkpoint.json

# View checkpoint
cat data/sft_production_v1/checkpoint.json
```

### Issue: Want to restart from scratch
**Solution:** Delete output directory or checkpoint
```bash
# Delete checkpoint only (keeps existing data)
rm data/sft_production_v1/checkpoint.json

# Or delete entire output directory
rm -rf data/sft_production_v1
```

---

## Summary

### Key Features:
- ✅ **Incremental saves** every N pairs (no data loss)
- ✅ **Full path resolution** (run from anywhere)
- ✅ **Auto-resume** from checkpoint on interruption
- ✅ **Error recovery** (catches API errors, continues)
- ✅ **Progress tracking** (detailed stats at each checkpoint)

### Recommended Command:

```bash
cd /Users/josmaiga/.claude-worktrees/sft/pedantic-hertz

python3 scripts/generate_sft_data_safe.py \
    --input-jsonl data_processed/medec_paired/train_val_split/sft_train.jsonl \
    --output-dir data/sft_production_v1 \
    --prompt-file configs/prompts/gpt4o_medec_safe_augmentation.json \
    --num-pairs 500 \
    --add-reasoning \
    --save-every 5 \
    --openai-api-key "$OPENAI_API_KEY"
```

**All paths will be printed at start - verify they're correct before generation begins!**
