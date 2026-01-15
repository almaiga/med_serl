# SFT Data Generation Guide for MedSeRL

## Overview

This guide covers generating high-quality SFT (Supervised Fine-Tuning) training data for MedSeRL using GPT-4o's advanced capabilities.

## Why SFT Training Data?

Base Qwen3-4B understands the task format but doesn't execute modifications correctly:
- ✅ Knows what should be changed
- ✅ Generates proper metadata
- ❌ **Doesn't actually apply the change to the note text**

SFT training teaches the model to:
1. **Actually modify** the clinical note text (not just describe changes)
2. Follow the output format consistently
3. Make appropriate medical changes

## Two Approaches

### Approach 1: Simple (For Small Models)
**File**: `configs/prompts/error_injection_prompts_v3_enhanced.json`
- Minimal changes (1-3 words in one sentence)
- Explicit preservation constraints
- Execution checklists
- **Best for**: Qwen3-4B (small model needs simple, clear instructions)

### Approach 2: Advanced (Leverage GPT-4o)
**File**: `configs/prompts/gpt4o_sft_generation_prompts.json`
- Complex paraphrasing (30-50% word changes)
- 15+ diverse error scenarios
- Optional complexity enhancement
- Rich metadata generation
- **Best for**: Creating diverse, challenging training data

## Generation Scripts

### Script 1: Basic Generation
**File**: `scripts/generate_sft_data.py`

**Features**:
- Conservative paraphrasing with entity preservation
- Multi-verifier consensus (Entity check + VCF + GPT-4o + Claude)
- MEDEC ground truth enhancement for errors

**Usage**:
```bash
python scripts/generate_sft_data.py \
    --input-jsonl data/medec_train.jsonl \
    --output-dir data/sft_training \
    --prompt-file configs/prompts/error_injection_prompts_v3_enhanced.json \
    --num-correct 500 \
    --num-incorrect 500 \
    --openai-api-key $OPENAI_API_KEY \
    --anthropic-api-key $ANTHROPIC_API_KEY  # Optional
```

### Script 2: Advanced Generation
**File**: `scripts/generate_sft_data_advanced.py`

**Features**:
- Ambitious paraphrasing (substantial rewrites)
- Diverse error scenarios (medication, lab, diagnostic, temporal, etc.)
- Optional complexity enhancement (adds clinical details)
- Multi-model verification
- Rich metadata (error severity, clinical impact, detection difficulty)

**Usage**:
```bash
python scripts/generate_sft_data_advanced.py \
    --input-jsonl data/medec_train.jsonl \
    --output-dir data/sft_training_advanced \
    --prompt-file configs/prompts/gpt4o_sft_generation_prompts.json \
    --num-correct 1000 \
    --num-incorrect 1000 \
    --enhance-complexity \
    --openai-api-key $OPENAI_API_KEY \
    --anthropic-api-key $ANTHROPIC_API_KEY  # Optional
```

## Verification Pipeline

### For CORRECT Paraphrases

**Stage 1: Entity Preservation (Fast, Deterministic)**
- Extracts: medications, dosages, measurements, laterality
- Verifies: All critical entities preserved exactly
- **Fails if**: Any medication name, dose, or measurement changed

**Stage 2: VCF Similarity (Fast, Deterministic)**
- Checks Jaccard similarity (0.85-0.99 range)
- Counts word edits (≤6 for simple, more flexible for advanced)
- **Fails if**: Too similar (no meaningful change) or too different

**Stage 3: GPT-4o Verification (Slow, Expensive)**
- Systematic comparison of medications, labs, diagnoses, etc.
- Classification: EQUIVALENT vs ERROR_DETECTED
- Confidence score (0.0-1.0)

**Stage 4: Claude Verification (Optional, Slow)**
- Independent verification with different model
- Reduces single-model bias
- Consensus voting with GPT-4o

**Consensus**: Require majority (≥50%) agreement + all deterministic checks pass

### For INCORRECT Examples

**Simpler verification**:
1. Parse error metadata from GPT-4o response
2. Verify note actually changed (not identical to original)
3. Apply VCF to ensure error is subtle (passes similarity thresholds)

## Error Diversity (Advanced Mode)

### 15+ Error Categories

1. **Medication Errors**:
   - Wrong drug (similar sounding, same class)
   - Wrong dose (10x overdose, underdose)
   - Wrong frequency (daily→weekly)
   - Wrong route (PO→IV)
   - Drug-drug interactions
   - Allergy violations

2. **Laboratory Errors**:
   - Wrong value (decimal point error)
   - Wrong unit (mg/dL→g/dL)
   - Wrong interpretation (elevated→normal)
   - Inconsistent with diagnosis

3. **Diagnostic Errors**:
   - Wrong diagnosis (similar presentation)
   - Wrong severity (mild→severe)
   - Wrong location/laterality
   - Missed diagnosis

4. **Temporal Errors**:
   - Wrong duration (days→months)
   - Wrong sequence (treatment before diagnosis)
   - Timeline inconsistencies

5. **Treatment Errors**:
   - Contraindicated treatment
   - Wrong procedure
   - Missing monitoring

6. **Contextual Errors**:
   - Age-inappropriate (pediatric drug in adult)
   - Pregnancy contraindications
   - Gender-specific issues

## Output Format

### Example CORRECT:
```json
{
  "original_note": "Patient has type 2 diabetes controlled with metformin 1000mg BID...",
  "generated_note": "Patient presents with type 2 diabetes mellitus managed on metformin 1000mg twice daily...",
  "label": "CORRECT",
  "metadata": {
    "generation_type": "correct_paraphrase",
    "enhanced": false,
    "changes_made": {
      "words_changed": "has → presents with, controlled → managed, BID → twice daily"
    }
  },
  "verification": {
    "gpt4o_verdict": {"classification": "EQUIVALENT", "confidence": 0.95},
    "claude_verdict": {"classification": "EQUIVALENT", "confidence": 0.92},
    "consensus": "EQUIVALENT",
    "confidence": 0.935
  }
}
```

### Example INCORRECT:
```json
{
  "original_note": "Started on furosemide 40mg daily and lisinopril 10mg daily...",
  "generated_note": "Started on furosemide 40mg daily and losartan 10mg daily...",
  "label": "INCORRECT",
  "metadata": {
    "generation_type": "incorrect_injection",
    "error_category": "medication",
    "subcategory": "wrong_drug_same_class",
    "severity": "moderate",
    "location": "medication_list",
    "original_text": "lisinopril 10mg daily",
    "modified_text": "losartan 10mg daily",
    "clinical_impact": "Switches ACE inhibitor to ARB without indication",
    "detection_difficulty": "moderate"
  }
}
```

## Recommended Workflow

### Phase 1: Generate Test Batch (50 examples)
```bash
# Test with small batch first
python scripts/generate_sft_data_advanced.py \
    --input-jsonl data/medec_train.jsonl \
    --output-dir data/sft_test \
    --num-correct 25 \
    --num-incorrect 25 \
    --openai-api-key $OPENAI_API_KEY
```

### Phase 2: Manual Review
- Review `data/sft_test/sft_combined_advanced.jsonl`
- Check CORRECT examples preserve all medical facts
- Check INCORRECT examples have realistic, detectable errors
- Verify output format is correct

### Phase 3: Generate Full Dataset (1000-2000 examples)
```bash
# Generate production dataset
python scripts/generate_sft_data_advanced.py \
    --input-jsonl data/medec_train.jsonl \
    --output-dir data/sft_production \
    --num-correct 1000 \
    --num-incorrect 1000 \
    --enhance-complexity \
    --openai-api-key $OPENAI_API_KEY \
    --anthropic-api-key $ANTHROPIC_API_KEY
```

### Phase 4: SFT Training
```bash
# Train Qwen3-4B on generated data
# (Use your existing SFT training script)
python src/training/train_serl.py \
    --phase sft \
    --model-name Qwen/Qwen3-4B \
    --train-data data/sft_production/sft_combined_advanced.jsonl \
    --save-path outputs/qwen3_4b_sft_checkpoint \
    --num-epochs 3 \
    --batch-size 4 \
    --learning-rate 5e-5
```

### Phase 5: Validate Trained Model
```bash
# Test trained model
python scripts/test_qwen3_batch_proper.py  # Should now work!
```

### Phase 6: MedSeRL RL Training
```bash
# Now ready for RL training
bash scripts/train_medserl_reinforce_pp.sh
```

## Cost Estimation

### GPT-4o Pricing (as of 2024):
- Input: ~$2.50 per 1M tokens
- Output: ~$10 per 1M tokens

### Estimated Costs:

**Per Example**:
- CORRECT (with verification): ~2,000 tokens = ~$0.025
- INCORRECT: ~1,500 tokens = ~$0.020

**For 1000 examples (500 CORRECT + 500 INCORRECT)**:
- CORRECT: 500 × $0.025 = ~$12.50
- INCORRECT: 500 × $0.020 = ~$10.00
- **Total: ~$22.50**

**For 2000 examples**:
- **Total: ~$45**

**With Claude verification** (adds ~$0.01 per CORRECT example):
- **Total for 1000: ~$27.50**

## Tips for Success

### 1. Start Small
- Generate 50-100 examples first
- Manually review quality
- Adjust prompts if needed

### 2. Balance Diversity
- Use `error_category` rotation for diverse errors
- Mix simple and complex notes
- Use `--enhance-complexity` for 30% of examples

### 3. Quality Over Quantity
- 500 high-quality examples > 2000 mediocre examples
- Strict verification prevents bad data
- Manuel review of failures helps improve prompts

### 4. Monitor Costs
- GPT-4o API calls add up
- Use `--num-correct` and `--num-incorrect` to control
- Consider caching/batching for efficiency

### 5. Iterate on Prompts
- If verification fails often, prompts may be too aggressive
- If examples too similar, increase temperature
- Check `generation_stats_advanced.json` for failure patterns

## Troubleshooting

### Issue: Low verification pass rate (<50%)
**Solution**:
- Reduce complexity in prompts
- Lower temperature (0.6 instead of 0.8)
- Add more explicit constraints

### Issue: Examples too similar to originals
**Solution**:
- Increase temperature (0.9-1.0)
- Emphasize diversity in prompts
- Use advanced mode with 30-50% word changes

### Issue: GPT-4o not following format
**Solution**:
- Use JSON mode (`response_format={"type": "json_object"}`)
- Add explicit output format examples in prompts
- Parse with error handling

### Issue: Verification too strict/lenient
**Solution**:
- Adjust VCF thresholds (min_jaccard, max_jaccard)
- Tune consensus threshold (currently ≥50%)
- Add/remove verifiers

## Next Steps After Generation

1. **Validate Generated Data**
   - Manual review of sample (50-100 examples)
   - Check for consistent quality
   - Verify format compatibility with training script

2. **SFT Training**
   - Train Qwen3-4B for 2-3 epochs
   - Monitor loss convergence
   - Validate on held-out test set

3. **Test Trained Model**
   - Run `test_qwen3_batch_proper.py`
   - Should now properly modify notes
   - VCF should start passing

4. **MedSeRL RL Training**
   - Now ready for full RL training
   - Model can execute note modifications
   - RL will improve quality further

## Summary

✅ **Simple approach** (v3_enhanced prompts): Conservative, explicit, good for small models
✅ **Advanced approach** (GPT-4o prompts): Ambitious, diverse, leverages GPT-4o's full capabilities
✅ **Multi-stage verification**: High confidence in data quality
✅ **Rich metadata**: Enables analysis and filtering
✅ **Cost-effective**: ~$20-50 for 1000-2000 examples

Ready to generate high-quality SFT data for MedSeRL! 🚀
