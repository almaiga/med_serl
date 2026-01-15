# Testing Interesting but Localized CORRECT Changes

## Setup Complete ✅

The prompt configuration has been updated with **6 interesting paraphrase strategies**:

1. **Reorder information** (same facts, different sequence)
2. **Add redundant context** (implied information made explicit)
3. **Medical/lay term swap** (same symptom, different terminology)
4. **Format change** (same data, different presentation)
5. **Temporal rephrasing** (same timeline, different expression)
6. **Unit conversion** (same value, different standard units)

## How to Test

### 1. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

### 2. Run test generation (2 pairs):
```bash
python3 scripts/generate_sft_data_safe.py \
    --input-jsonl data_processed/medec_paired/train_val_split/sft_train.jsonl \
    --output-dir data/sft_interesting_changes_test \
    --prompt-file configs/prompts/gpt4o_medec_safe_augmentation.json \
    --num-pairs 2 \
    --add-reasoning \
    --openai-api-key "$OPENAI_API_KEY"
```

### 3. Inspect the output:
```bash
# View CORRECT examples with reasoning
cat data/sft_interesting_changes_test/sft_correct.jsonl | jq '.'

# Check if changes are INTERESTING (not trivial like "lives→resides")
cat data/sft_interesting_changes_test/sft_correct.jsonl | jq '.reasoning.change_made'

# Verify they're LOCALIZED (not whole note rewrites)
cat data/sft_interesting_changes_test/sft_correct.jsonl | jq '.reasoning.paraphrase_technique'

# View INCORRECT examples with injector reasoning
cat data/sft_interesting_changes_test/sft_incorrect.jsonl | jq '.reasoning.injection_strategy'
```

## Expected Quality Criteria

### ✅ CORRECT Changes Should Be:

**INTERESTING** (require model reasoning to verify):
- ❌ BAD: "lives" → "resides" (trivial synonym)
- ❌ BAD: "note" → "observe" (trivial synonym)
- ✅ GOOD: "Patient has diabetes, on metformin 1000mg BID" → "On metformin 1000mg BID for diabetes management" (reordering)
- ✅ GOOD: "shortness of breath" → "dyspnea" (medical/lay swap)
- ✅ GOOD: "BP 140/90" → "Blood pressure measured at 140/90 mmHg" (format change)
- ✅ GOOD: "3 days ago" → "72 hours prior to presentation" (temporal rephrase)

**LOCALIZED** (verifiable change, not whole note rewrite):
- ✅ Change 1 sentence or 1 section
- ✅ Rest of note stays identical
- ✅ Easy to verify medical equivalence

**SAFE** (preserves all medical facts):
- ✅ Same diagnosis
- ✅ Same medications (exact drug, dose, frequency)
- ✅ Same lab values (can change units if standard)
- ✅ Same timeline (can change expression)

### ✅ INCORRECT Changes Should Have:

**Injector Reasoning** explaining:
- `injection_strategy`: What specific change was made
- `why_plausible`: Why this error looks realistic
- `deception_technique`: What makes it subtle/hard to detect
- `clinical_reasoning`: Why this error might occur in practice
- `detection_clues`: What assessor needs to catch it
- `error_type_pattern`: General pattern for this error type

## Example Expected Output

### CORRECT Example:
```json
{
  "note": "A 42-year-old woman is brought to the physician by her husband because of a 1-year history of abnormal behavior. Throughout this period she has been irritable, restless, and has had multiple episodes of hearing voices...",
  "label": "CORRECT",
  "original_note_id": "ms-train-418",
  "source": "medec_correct_paraphrased",
  "reasoning": {
    "change_made": "Changed 'During this time' to 'Throughout this period' in the second sentence",
    "why_safe": "Both phrases refer to the same 1-year duration mentioned in the previous sentence. 'During this time' and 'throughout this period' are temporally equivalent.",
    "preserved_facts": [
      "1-year history of abnormal behavior",
      "Irritability, restlessness, and auditory hallucinations",
      "All other clinical details unchanged"
    ],
    "paraphrase_technique": "Temporal rephrasing - same duration, different expression",
    "medical_equivalence": "Both 'during this time' and 'throughout this period' refer to the same 1-year timeframe, maintaining exact temporal continuity"
  }
}
```

### INCORRECT Example:
```json
{
  "note": "A 42-year-old woman... Suspected of Creutzfeldt-Jakob disease...",
  "label": "INCORRECT",
  "original_note_id": "ms-train-418",
  "error_type": "diagnosis",
  "error_location": "Suspected of Creutzfeldt-Jakob disease.",
  "correction": "Suspected of Huntington disease.",
  "source": "medec_incorrect_verified",
  "reasoning": {
    "injection_strategy": "Changed diagnosis from 'Huntington disease' to 'Creutzfeldt-Jakob disease'",
    "why_plausible": "Both are neurodegenerative diseases causing movement disorders, cognitive decline, and psychiatric symptoms - easy to confuse without careful analysis",
    "deception_technique": "The symptoms overlap significantly: both have abnormal movements, behavioral changes, and dementia. Without checking family history closely, they appear similar.",
    "clinical_reasoning": "This error occurs in practice because both are rare, both affect movement and cognition, and clinicians may anchor on prominent symptoms without fully considering timeline and genetics.",
    "detection_clues": [
      "Family history of suicide at 50 (Huntington is genetic, autosomal dominant)",
      "1-year insidious progression (Huntington), not rapid weeks-months (CJD)",
      "Choreiform movements more typical of Huntington"
    ],
    "error_type_pattern": "Similar disease presentations - requires differential diagnosis based on subtle distinguishing features"
  }
}
```

## Quality Checks

After generation, verify:

### For CORRECT:
- [ ] Changes are NOT trivial synonyms like "lives→resides"
- [ ] Changes require model REASONING to verify safety
- [ ] Changes are LOCALIZED (1 sentence or section)
- [ ] All medical facts preserved exactly
- [ ] Reasoning explains why change is safe

### For INCORRECT:
- [ ] Identical to MEDEC incorrect_note (run `diff`)
- [ ] Reasoning explains injector strategy
- [ ] Reasoning highlights what makes error plausible
- [ ] Detection clues provided

## Production Generation

Once test output looks good (2 pairs verified), scale to production:

```bash
python3 scripts/generate_sft_data_safe.py \
    --input-jsonl data_processed/medec_paired/train_val_split/sft_train.jsonl \
    --output-dir data/sft_production \
    --prompt-file configs/prompts/gpt4o_medec_safe_augmentation.json \
    --num-pairs 500 \
    --add-reasoning \
    --openai-api-key "$OPENAI_API_KEY"
```

**Expected cost:** ~$20-25 for 500 pairs (1000 examples)
**Expected time:** ~1-2 hours with rate limiting

## Summary of Changes Made

### Updated Files:
1. **`configs/prompts/gpt4o_medec_safe_augmentation.json`**
   - Added 6 interesting paraphrase strategies to `correct_paraphrase_system`
   - Increased temperature from 0.5 to 0.7 for more creative paraphrasing
   - Emphasized goal: "Make model THINK to verify it's safe"
   - Added examples for each strategy

2. **`scripts/generate_sft_data_safe.py`**
   - Already has dual reasoning implementation (CORRECT + INCORRECT)
   - Tracks reasoning stats separately for each type

3. **Documentation**
   - Created `INTERESTING_CORRECT_CHANGES.md` with detailed examples
   - Created this test guide

### Key Principles:
- ✅ **Interesting**: Not trivial synonyms - require reasoning to verify
- ✅ **Localized**: Change 1 sentence/section, not whole note
- ✅ **Safe**: Preserve all medical facts exactly
- ✅ **Verifiable**: Easy to confirm medical equivalence
- ✅ **Dual Reasoning**: Both CORRECT and INCORRECT have explanations

The system is now ready to generate interesting but safe CORRECT changes that will actually train the model to verify medical equivalence!
