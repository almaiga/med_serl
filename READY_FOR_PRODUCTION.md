# Ready for Production: SFT Data Generation

## ✅ Implementation Complete

All components are ready for production data generation:

### 1. ✅ Prompt Configuration
**File:** `configs/prompts/gpt4o_medec_safe_augmentation.json`

**Features:**
- ✅ 5 realistic strategies for CORRECT notes (temporal, terminology, anatomical, unit conversion, medication context)
- ✅ Single change enforcement (localized, verifiable)
- ✅ Chain-of-thought reasoning for both CORRECT and INCORRECT
- ✅ Clear verification chain (4 steps for CORRECT)
- ✅ Clear deception chain (4 steps for INCORRECT)

### 2. ✅ Generation Script
**File:** `scripts/generate_sft_data_safe.py`

**Features:**
- ✅ Dual reasoning (CORRECT + INCORRECT)
- ✅ MEDEC errors used AS-IS (doctor-verified)
- ✅ GPT-4o for CORRECT paraphrasing only
- ✅ 50/50 output split
- ✅ Stats tracking with separate reasoning counts

### 3. ✅ Documentation
**Files created:**
- ✅ `REALISTIC_CORRECT_STRATEGIES.md` - Detailed strategy guide
- ✅ `FINAL_SFT_DATA_STRATEGY.md` - Complete training strategy
- ✅ `INTERESTING_CORRECT_CHANGES.md` - Examples and rationale
- ✅ `INJECTOR_REASONING_GUIDE.md` - Injector perspective explanation
- ✅ `TEST_OUTPUT_ANALYSIS.md` - Quality analysis from tests

---

## Production Generation Command

### Final Test (2 pairs to verify CoT reasoning)
```bash
python3 scripts/generate_sft_data_safe.py \
    --input-jsonl data_processed/medec_paired/train_val_split/sft_train.jsonl \
    --output-dir data/sft_final_test \
    --prompt-file configs/prompts/gpt4o_medec_safe_augmentation.json \
    --num-pairs 2 \
    --add-reasoning \
    --openai-api-key "$OPENAI_API_KEY"
```

**Check output for:**
- [ ] CORRECT has `verification_chain` with 4 steps
- [ ] INCORRECT has `deception_chain` with 4 steps
- [ ] Only ONE change per CORRECT note
- [ ] Strategy explicitly labeled (Strategy 1-5)

### Production Run (500 pairs = 1000 examples)
```bash
python3 scripts/generate_sft_data_safe.py \
    --input-jsonl data_processed/medec_paired/train_val_split/sft_train.jsonl \
    --output-dir data/sft_production_v1 \
    --prompt-file configs/prompts/gpt4o_medec_safe_augmentation.json \
    --num-pairs 500 \
    --add-reasoning \
    --openai-api-key "$OPENAI_API_KEY"
```

**Expected:**
- Output: 1000 examples (500 CORRECT + 500 INCORRECT)
- Cost: ~$25-35 (with chain-of-thought reasoning)
- Time: 1-2 hours with rate limiting
- Files:
  - `sft_correct.jsonl` (500 CORRECT with verification chains)
  - `sft_incorrect.jsonl` (500 INCORRECT with deception chains)
  - `sft_combined_safe.jsonl` (1000 total, shuffled 50/50)
  - `generation_stats_safe.json` (detailed statistics)

---

## Expected Output Format

### CORRECT Example (with CoT):
```json
{
  "note": "A 42-year-old woman is brought to the physician by her husband because of a 12-month history of abnormal behavior...",
  "label": "CORRECT",
  "original_note_id": "ms-train-418",
  "source": "medec_correct_paraphrased",
  "metadata": {
    "generation_type": "correct_paraphrase",
    "gpt4o_tokens": 1800
  },
  "reasoning": {
    "change_made": "Changed '1-year history' to '12-month history'",
    "verification_chain": [
      "Step 1: Only one temporal phrase changed - '1-year history' → '12-month history', rest identical",
      "Step 2: Check equivalence - 1 year = 12 months (mathematical equivalence)",
      "Step 3: Verify facts preserved - patient age (42), symptoms (abnormal behavior), all other details unchanged",
      "Step 4: Strategy 1 (Temporal Equivalence) applied correctly - same duration, different units"
    ],
    "why_safe": "1 year and 12 months are mathematically equivalent. All other clinical facts preserved exactly.",
    "strategy_used": "Strategy 1: TEMPORAL EQUIVALENCE",
    "preserved_facts": ["42-year-old patient", "Abnormal behavior symptoms", "All other timeline details unchanged"]
  }
}
```

### INCORRECT Example (with CoT):
```json
{
  "note": "...Suspected of Creutzfeldt-Jakob disease...",
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
    "injection_strategy": "Changed 'Huntington disease' to 'Creutzfeldt-Jakob disease'",
    "deception_chain": [
      "Step 1: Found similar disease - both neurodegenerative with movement disorders and cognitive decline",
      "Step 2: Verified symptom overlap - confusion, abnormal movements, psychiatric symptoms fit both",
      "Step 3: Located subtle differentiator - family history (genetic Huntington vs sporadic CJD) and timeline (1-year insidious vs rapid weeks-months)",
      "Step 4: Bet assessor won't check carefully - family history buried in social history section, easy to miss genetic pattern"
    ],
    "why_plausible": "Both are rare neurodegenerative diseases with overlapping symptoms (movement disorders, cognitive decline, psychiatric features). Without careful family history analysis, they appear similar.",
    "detection_requires": "Assessor must: (1) recognize family history pattern (father's suicide at 50 suggests genetic disease), (2) check timeline (1-year progression too slow for CJD), (3) note specific movement type (choreiform movements more typical of Huntington)",
    "detection_clues": [
      "Father committed suicide at 50 - suggests genetic autosomal dominant condition (Huntington)",
      "1-year insidious progression - too slow for CJD (usually weeks-months)",
      "Choreiform movements - more characteristic of Huntington than CJD"
    ],
    "error_type_pattern": "Similar disease presentations requiring differential diagnosis"
  }
}
```

---

## Quality Verification Checklist

After generation, verify:

### CORRECT Examples:
- [ ] Each note has EXACTLY ONE change (run diff vs original)
- [ ] `verification_chain` has 4 steps
- [ ] `strategy_used` is explicitly labeled (Strategy 1-5)
- [ ] `preserved_facts` lists critical unchanged elements
- [ ] Change is medically safe (verify no fact alterations)
- [ ] Change is realistic (happens in real clinical practice)

### INCORRECT Examples:
- [ ] Note is IDENTICAL to MEDEC incorrect_note (run diff)
- [ ] `deception_chain` has 4 steps
- [ ] `detection_requires` explains assessor strategy
- [ ] `detection_clues` lists 3+ specific clues
- [ ] `error_type_pattern` identifies general pattern

### Statistics:
- [ ] 50/50 split (500 CORRECT, 500 INCORRECT)
- [ ] All CORRECT have reasoning (500/500)
- [ ] All INCORRECT have reasoning (500/500)
- [ ] No generation failures
- [ ] Token usage tracked

---

## Next Steps After Generation

### 1. Quality Spot Check
```bash
# View random CORRECT examples
shuf data/sft_production_v1/sft_correct.jsonl | head -5 | jq '.'

# View random INCORRECT examples
shuf data/sft_production_v1/sft_incorrect.jsonl | head -5 | jq '.'

# Check strategy distribution
cat data/sft_production_v1/sft_correct.jsonl | jq -r '.reasoning.strategy_used' | sort | uniq -c

# Check error type distribution
cat data/sft_production_v1/sft_incorrect.jsonl | jq -r '.error_type' | sort | uniq -c
```

### 2. Manual Verification (Sample 10-20)
- [ ] Verify CORRECT changes are single and safe
- [ ] Verify INCORRECT notes are MEDEC AS-IS
- [ ] Check reasoning chains are coherent
- [ ] Confirm strategies are correctly applied

### 3. SFT Training
```bash
python src/training/train_serl.py \
    --phase sft \
    --model-name Qwen/Qwen3-4B \
    --train-data data/sft_production_v1/sft_combined_safe.jsonl \
    --save-path outputs/qwen3_4b_sft_v1 \
    --num-epochs 3 \
    --batch-size 4 \
    --learning-rate 5e-5
```

### 4. RL Self-Play Training
```bash
# After SFT completes
bash scripts/train_medserl_reinforce_pp.sh
```

---

## Summary: What We Built

### Data Generation:
- ✅ **5 realistic strategies** for CORRECT notes (based on real clinical practice)
- ✅ **MEDEC errors AS-IS** for INCORRECT notes (doctor-verified)
- ✅ **Chain-of-thought reasoning** for both types (teaches HOW to verify/inject)
- ✅ **Single change enforcement** (localized, verifiable)
- ✅ **50/50 split** (balanced training data)

### Training Goals:
1. **SFT:** Teach task format + reasoning patterns
2. **RL:** Drive creative exploration + adversarial improvement

### Model Capabilities:
1. **Injector role:** Generate plausible medical errors strategically
2. **Assessor role:** Detect errors through systematic verification
3. **Chain-of-thought:** Reason step-by-step during rollouts
4. **Strategy awareness:** Apply learned patterns to novel cases

**Key Innovation:** Chain-of-thought reasoning teaches the MODEL's thought process, not just input-output mappings!

---

## Ready to Generate! 🚀

All components are in place. You can now:
1. Run final test (2 pairs) to verify CoT reasoning
2. Review test output for quality
3. Generate production data (500 pairs)
4. Begin SFT training

The system is designed to teach both VERIFICATION (assessor) and DECEPTION (injector) through explicit reasoning chains!
