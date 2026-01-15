# Safe SFT Data Generation Guide

## Overview

This guide covers generating **safe, doctor-verified** SFT training data for MedSeRL using the MEDEC dataset and GPT-4o.

## Key Principles

### ✅ What We Do

1. **Use MEDEC errors AS-IS** - Already doctor-verified, no risk
2. **Generate CORRECT paraphrases only** - Safe, no medical risk
3. **50/50 CORRECT/INCORRECT split** - Prevents model bias
4. **Optional reasoning** - GPT-4o explains WHY errors are wrong (helps small model learn)

### ❌ What We DON'T Do

1. **Never generate new errors** - Too risky without MD verification
2. **Never modify MEDEC incorrect_note** - Trust the curation
3. **Never create unbalanced datasets** - Maintain 50/50 calibration

## Why This Approach?

### Safety First

- **Medical errors are risky**: Generated errors could teach model dangerous patterns
- **Doctor verification is expensive**: MEDEC already has verified error pairs
- **Trust the experts**: Use existing medical curation, don't reinvent

### Effectiveness

- **GPT-4o for CORRECT**: Leverages strong reasoning for paraphrasing
- **MEDEC for INCORRECT**: Uses real clinical error patterns
- **50/50 calibration**: Model learns to detect errors without bias
- **Educational reasoning**: Small model benefits from GPT-4o's explanations

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MEDEC Dataset                            │
│  (Doctor-verified error pairs)                              │
│                                                              │
│  {                                                           │
│    "note_id": "ms-train-418",                               │
│    "correct_note": "...Huntington disease...",              │
│    "incorrect_note": "...Creutzfeldt-Jakob disease...",     │
│    "error_type": "diagnosis"                                │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────┐
         │   Safe SFT Data Generation     │
         │   (scripts/generate_sft_data_safe.py) │
         └────────────────────────────────┘
                          │
         ┌────────────────┴────────────────┐
         │                                  │
         ▼                                  ▼
┌─────────────────┐              ┌──────────────────┐
│ CORRECT Branch  │              │ INCORRECT Branch │
│ (GPT-4o)        │              │ (MEDEC AS-IS)    │
└─────────────────┘              └──────────────────┘
         │                                  │
         │ Paraphrase                       │ Use AS-IS
         │ correct_note                     │ incorrect_note
         │                                  │
         ▼                                  ▼
┌─────────────────┐              ┌──────────────────┐
│ {               │              │ {                │
│  "note": "...", │              │  "note": "...",  │
│  "label": "CORRECT",           │  "label": "INCORRECT",
│  "source": "medec_correct_paraphrased" │  "source": "medec_incorrect_verified"
│ }               │              │  "reasoning": {...} # Optional
┌─────────────────┘              │ }                │
         │                       └──────────────────┘
         │                                  │
         └──────────────┬───────────────────┘
                        ▼
         ┌──────────────────────────────┐
         │  Combined SFT Dataset        │
         │  (50% CORRECT / 50% INCORRECT) │
         │                              │
         │  sft_combined_safe.jsonl     │
         └──────────────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │   SFT Training (Qwen3-4B)    │
         │   Learns to execute mods     │
         └──────────────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │   RL Self-Play Training      │
         │   (Injector vs Assessor)     │
         └──────────────────────────────┘
```

## Step-by-Step Workflow

### Step 1: Prepare Environment

```bash
# Install dependencies
pip install openai tqdm

# Set OpenAI API key
export OPENAI_API_KEY=your-key-here

# Verify MEDEC data exists
ls data_processed/medec_paired/train_val_split/sft_train.jsonl
```

### Step 2: Test with Small Batch (10 pairs)

```bash
# Quick test to verify everything works
python scripts/test_safe_generation.py
```

**Expected output:**
```
Generating safe SFT data...
Strategy: Use MEDEC errors AS-IS, generate CORRECT paraphrases
Output: 50/50 CORRECT/INCORRECT split

Generating SFT data: 100%|████████████████| 10/10

Safe SFT Data Generation Complete
Total pairs processed: 10
CORRECT paraphrases generated: 10
INCORRECT notes used (AS-IS): 10
Reasoning added: 10
Total GPT-4o tokens used: ~25,000

Verification Report
Total examples: 20
  CORRECT: 10 (50.0%)
  INCORRECT: 10 (50.0%)

✅ Verification complete
```

### Step 3: Manual Quality Review

Review the test output in `data/sft_test_safe/`:

```bash
# Check CORRECT paraphrases
head -n 1 data/sft_test_safe/sft_correct.jsonl | jq .

# Check INCORRECT notes (should be MEDEC AS-IS)
head -n 1 data/sft_test_safe/sft_incorrect.jsonl | jq .

# Check reasoning (if added)
head -n 1 data/sft_test_safe/sft_incorrect.jsonl | jq .reasoning
```

**Quality checklist for CORRECT paraphrases:**
- [ ] All medications preserved exactly (names, doses, routes, frequencies)
- [ ] All lab values preserved exactly (numbers, units)
- [ ] All diagnoses preserved (can use synonyms)
- [ ] Natural clinical writing (not obviously AI-generated)
- [ ] Substantial variation from original (not just copy-paste)

**Quality checklist for INCORRECT notes:**
- [ ] Identical to MEDEC incorrect_note (verify with diff)
- [ ] Error location correctly identified
- [ ] Reasoning explains WHY error is wrong (if added)

### Step 4: Generate Full Dataset

Once quality checks pass, generate full dataset:

```bash
# Generate 500 CORRECT + 500 INCORRECT (1000 total examples)
python scripts/generate_sft_data_safe.py \
    --input-jsonl data_processed/medec_paired/train_val_split/sft_train.jsonl \
    --output-dir data/sft_medec_safe_production \
    --prompt-file configs/prompts/gpt4o_medec_safe_augmentation.json \
    --num-pairs 500 \
    --add-reasoning \
    --openai-api-key $OPENAI_API_KEY
```

**Output:**
```
data/sft_medec_safe_production/
├── sft_correct.jsonl              # 500 CORRECT paraphrases
├── sft_incorrect.jsonl            # 500 INCORRECT (MEDEC AS-IS)
├── sft_combined_safe.jsonl        # 1000 examples (50/50 split)
└── generation_stats_safe.json     # Statistics
```

**Cost estimate for 500 pairs:**
- CORRECT paraphrases: 500 × ~2,500 tokens = ~1.25M tokens
- Reasoning generation: 500 × ~1,500 tokens = ~750K tokens
- **Total: ~2M tokens ≈ $25-30**

### Step 5: Validate Generated Dataset

```bash
# Run validation script
python scripts/validate_sft_data.py \
    --input-jsonl data/sft_medec_safe_production/sft_combined_safe.jsonl \
    --output-report data/sft_medec_safe_production/validation_report.json
```

**Validation checks:**
- Label distribution (should be 50/50)
- Source verification (CORRECT from paraphrase, INCORRECT from MEDEC)
- Format consistency
- No duplicate note_ids

### Step 6: SFT Training

Train Qwen3-4B on the generated data:

```bash
# Use your existing SFT training script
python src/training/train_serl.py \
    --phase sft \
    --model-name Qwen/Qwen3-4B \
    --train-data data/sft_medec_safe_production/sft_combined_safe.jsonl \
    --save-path outputs/qwen3_4b_sft_safe \
    --num-epochs 3 \
    --batch-size 4 \
    --learning-rate 5e-5
```

### Step 7: Test Trained Model

Verify the model now executes modifications:

```bash
# Test with original test script
python scripts/test_qwen3_batch_proper.py

# Expected: VCF should now PASS (model applies modifications)
```

### Step 8: Proceed to RL Training

Once SFT model executes modifications correctly:

```bash
# Run full MedSeRL RL self-play training
bash scripts/train_medserl_reinforce_pp.sh
```

## Output Format

### CORRECT Example

```json
{
  "note": "A 42-year-old woman presents with a one-year history of behavioral changes...",
  "label": "CORRECT",
  "original_note_id": "ms-train-418",
  "source": "medec_correct_paraphrased",
  "metadata": {
    "generation_type": "correct_paraphrase",
    "gpt4o_tokens": 2450
  },
  "reasoning": null
}
```

### INCORRECT Example (with reasoning)

```json
{
  "note": "A 42-year-old woman... Suspected of Creutzfeldt-Jakob disease...",
  "label": "INCORRECT",
  "original_note_id": "ms-train-418",
  "error_type": "diagnosis",
  "error_location": "Suspected of Creutzfeldt-Jakob disease.",
  "correction": "Suspected of Huntington disease.",
  "source": "medec_incorrect_verified",
  "metadata": {
    "generation_type": "medec_verified_error"
  },
  "reasoning": {
    "error_identified": "Misdiagnosis: CJD instead of Huntington",
    "why_wrong": "Family history of father suicide at 50 suggests genetic Huntington (autosomal dominant, typical onset 30s-50s). Choreiform movements and 1-year insidious progression fit Huntington, not CJD (which is rapid, weeks-months).",
    "key_clues": [
      "Father suicide at 50 (Huntington genetic)",
      "1-year progression (Huntington insidious, CJD rapid)",
      "Choreiform movements (Huntington)"
    ],
    "correct_answer": "Huntington disease: autosomal dominant, chorea + psychiatric + cognitive decline, genetic family history",
    "clinical_impact": "Wrong diagnosis affects genetic counseling, family screening, prognosis"
  }
}
```

## Prompt Configuration

### Key Prompts (from `gpt4o_medec_safe_augmentation.json`)

**CORRECT Paraphrase System Prompt:**
```
You are a clinical documentation expert creating semantically equivalent paraphrases.

MUST Preserve Exactly:
- All numerical values (labs, vitals, ages, doses, measurements, units)
- All medications (drug names, doses, routes, frequencies)
- All diagnoses and clinical assessments

Can Vary:
- Sentence structure and word order
- Medical terminology (use synonyms)
- Documentation style (telegraphic ↔ full sentences)

Output: Only the paraphrased note. No explanations.
```

**Reasoning System Prompt:**
```
You are a medical educator explaining clinical errors.

Task: Given a clinical note pair (correct vs incorrect), explain:
1. What the error is
2. Why it's wrong (clinical reasoning)
3. Key clues to detect it
4. Clinical impact

Output JSON with: error_identified, why_wrong, key_clues, correct_answer, clinical_impact
```

## Troubleshooting

### Issue: Low paraphrase quality

**Symptoms**: CORRECT paraphrases too similar to original or changed medical facts

**Solutions:**
1. Increase temperature (0.7 → 0.9)
2. Add more examples in prompt
3. Manually review and filter bad examples

### Issue: Reasoning generation fails

**Symptoms**: `reasoning_failed` count is high

**Solutions:**
1. Check error in logs
2. Verify MEDEC pair has error_sentence and corrected_sentence
3. Use `response_format={"type": "json_object"}` for JSON output

### Issue: API rate limits

**Symptoms**: `RateLimitError` from OpenAI

**Solutions:**
1. Add `time.sleep(0.5)` between requests
2. Process in smaller batches
3. Use GPT-4o-mini for cheaper option (adjust prompt_file)

### Issue: Cost too high

**Symptoms**: Budget concerns for large datasets

**Solutions:**
1. Start with 100-200 pairs instead of 500
2. Skip reasoning generation (`--no-reasoning`)
3. Use GPT-4o-mini (cheaper but lower quality)
4. Generate incrementally over multiple days

## Cost Analysis

### Token Usage per Example

- **CORRECT paraphrase**: ~2,500 tokens (400 input + 600 output + 1,500 verification)
- **Reasoning generation**: ~1,500 tokens (400 input + 300 output)

### Pricing (GPT-4o as of 2024)

- Input: $2.50 per 1M tokens
- Output: $10 per 1M tokens

### Cost Estimates

| Dataset Size | CORRECT | INCORRECT | Reasoning | Total Cost |
|--------------|---------|-----------|-----------|------------|
| 100 pairs    | 100     | 100       | Optional  | ~$5        |
| 500 pairs    | 500     | 500       | Optional  | ~$25       |
| 1000 pairs   | 1000    | 1000      | Optional  | ~$50       |

**Note**: INCORRECT notes are free (MEDEC AS-IS), only CORRECT paraphrases and reasoning cost tokens.

## Quality Metrics

### Success Criteria

- **CORRECT paraphrases**: 90%+ entity preservation, natural variation
- **INCORRECT notes**: 100% identical to MEDEC (verified with diff)
- **Reasoning quality**: Clear explanation, 3+ key clues, clinical impact
- **Label balance**: 50.0% CORRECT, 50.0% INCORRECT (±1%)

### Manual Review Checklist

For each batch of 10-20 examples:

**CORRECT:**
- [ ] All medications exactly preserved?
- [ ] All lab values exactly preserved?
- [ ] Diagnosis preserved (synonyms OK)?
- [ ] Natural clinical writing?
- [ ] Substantial variation (not copy-paste)?

**INCORRECT:**
- [ ] Identical to MEDEC incorrect_note? (run diff)
- [ ] Error location correct?
- [ ] Reasoning explains WHY wrong?
- [ ] Clinical impact stated?

## Next Steps After Generation

1. **Validate dataset**: Run format checks, label distribution
2. **SFT training**: Train Qwen3-4B for 2-3 epochs
3. **Test execution**: Verify model applies modifications (VCF passes)
4. **RL training**: Run full MedSeRL self-play
5. **Evaluate**: Test on MEDEC test sets

## Summary

✅ **Safe approach**: Uses doctor-verified MEDEC errors, only generates CORRECT paraphrases
✅ **Effective**: 50/50 split for calibration, GPT-4o reasoning helps small model learn
✅ **Cost-efficient**: ~$25-50 for 500-1000 pairs
✅ **Production-ready**: Tested workflow with quality checks

**Key takeaway**: By using MEDEC errors AS-IS and only generating CORRECT paraphrases, we avoid the risk of training on bad medical data while still leveraging GPT-4o's capabilities for paraphrasing and reasoning.
