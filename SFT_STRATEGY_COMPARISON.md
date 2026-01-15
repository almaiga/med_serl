# SFT Data Generation Strategy Comparison

## Two Approaches for MedSeRL SFT Training

### Approach 1: Safe Augmentation (Conservative)
**File:** `configs/prompts/gpt4o_medec_safe_augmentation.json`
**Script:** `scripts/generate_sft_data_safe.py`

### Approach 2: Competitive Self-Play (Adversarial)
**File:** `configs/prompts/gpt4o_competitive_selfplay.json`
**Script:** `scripts/generate_competitive_sft.py`

---

## The Key Problem: Memorization vs. Learning

### What Happens Without Competition?

```
Training Round 1:
  Note: "Patient has type 2 diabetes on metformin 1000mg BID..."
  Model: "Hmm, I need to check medications..."
  → Classifies correctly

Training Round 10:
  Same Note: "Patient has type 2 diabetes on metformin 1000mg BID..."
  Model: "I've seen this exact text before → CORRECT"
  → Memorized, not learned

Self-play degenerates into text matching, not error detection.
```

---

## Approach 1: Safe Augmentation

### Strategy

**CORRECT Examples:**
- Paraphrase `correct_note` from MEDEC
- Preserve all medical facts exactly

**INCORRECT Examples:**
- Use MEDEC `incorrect_note` AS-IS (doctor-verified)
- Optional: Add reasoning explaining WHY error is wrong

### Pros

✅ **Safest**: No risk of generating bad medical errors
✅ **Doctor-verified**: All errors are from MEDEC curation
✅ **Clear reasoning**: GPT-4o explains errors (educational)
✅ **Simple**: Straightforward implementation

### Cons

❌ **Memorization risk**: Model may learn specific note text
❌ **Static errors**: MEDEC errors don't vary
❌ **Self-play degeneracy**: After SFT, RL self-play may become trivial
❌ **Not adversarial**: No competitive pressure

### When to Use

- **Initial SFT warmup**: Teach basic task format
- **Conservative training**: When medical safety is paramount
- **Limited data**: When you only have small MEDEC subset
- **Baseline**: To establish minimum performance

### Example Output

**CORRECT:**
```
Original: "Patient has type 2 diabetes controlled with metformin 1000mg BID."
Generated: "Patient presents with type 2 diabetes mellitus managed on metformin 1000mg twice daily."
```

**INCORRECT (MEDEC AS-IS):**
```
Original: "Started on lisinopril 10mg daily."
MEDEC Error: "Started on losartan 10mg daily."
Generated: "Started on losartan 10mg daily." [Exact copy from MEDEC]
+ Reasoning: "ACE-i → ARB switch without indication..."
```

---

## Approach 2: Competitive Self-Play

### Strategy

**CORRECT Examples:**
- Heavy paraphrase (30-50% word changes)
- Goal: Prevent memorization

**INCORRECT Examples:**
- Heavy paraphrase + apply error VARIATION
- Use MEDEC error TYPE/PATTERN but with different specifics
- Example: MEDEC has "lisinopril → losartan", generate "enalapril → valsartan"

### Pros

✅ **Prevents memorization**: Both CORRECT and INCORRECT are paraphrased
✅ **Competitive**: Keeps self-play adversarial
✅ **Pattern learning**: Model learns ERROR TYPES, not specific examples
✅ **Scalable**: Can generate unlimited variations
✅ **RL-ready**: Smooth transition from SFT to self-play RL

### Cons

❌ **More complex**: Requires error variation logic
❌ **Higher cost**: More GPT-4o calls for variations
❌ **Verification needed**: Must ensure error variations are still realistic
❌ **Potential drift**: Variations might deviate from MEDEC patterns

### When to Use

- **Full training pipeline**: SFT → RL self-play
- **Large-scale training**: 500+ pairs, multiple epochs
- **Competitive self-play**: When assessor needs to stay challenged
- **Production system**: When model must generalize to novel errors

### Example Output

**CORRECT:**
```
Original: "Patient has type 2 diabetes controlled with metformin 1000mg BID."
Generated: "Metformin 1000mg twice daily is being used to manage the patient's type 2 diabetes mellitus."
```

**INCORRECT (Paraphrased + Variation):**
```
Original: "Patient on lisinopril 10mg daily."
MEDEC Error Type: ACE-i → ARB confusion
Generated: "Patient is currently taking enalapril 10mg daily and blood pressure remains elevated. Switched to valsartan 10mg daily."
                                                                    ↑ Error variation (same class, different drug)
```

---

## Side-by-Side Comparison

| Feature | Safe Augmentation | Competitive Self-Play |
|---------|-------------------|----------------------|
| **CORRECT notes** | Light paraphrase | Heavy paraphrase (30-50%) |
| **INCORRECT notes** | MEDEC AS-IS | Paraphrase + error variation |
| **Error source** | Doctor-verified (MEDEC) | Pattern-based variation |
| **Memorization risk** | High | Low |
| **Self-play readiness** | Degenerates | Stays competitive |
| **Medical safety** | Highest | High (pattern-based) |
| **Cost (500 pairs)** | ~$25 | ~$35-40 |
| **Complexity** | Simple | Moderate |
| **Use case** | SFT warmup | SFT + RL pipeline |

---

## Recommendation

### For Your MedSeRL Training

**Use Competitive Self-Play** because:

1. **You're doing RL self-play**: Need competitive data from the start
2. **Prevent early degeneracy**: Model won't just memorize text
3. **Pattern learning**: Model learns to detect error TYPES (medication confusion, laterality, etc.)
4. **Smooth transition**: SFT model is already RL-ready

### Hybrid Approach (Best of Both)

You can combine both strategies:

**Phase 1: Safe Augmentation (50-100 pairs)**
- Initial SFT warmup
- Teach basic task format
- Use MEDEC errors AS-IS for safety

**Phase 2: Competitive Self-Play (400-500 pairs)**
- Scale up with variations
- Prevent memorization
- Prepare for RL training

**Phase 3: RL Self-Play**
- Injector learns adversarially
- Assessor stays challenged
- Convergence to realistic errors

---

## Implementation

### Test Safe Augmentation
```bash
python scripts/test_safe_generation.py
# Already working! ✅
```

### Test Competitive Self-Play
```bash
python scripts/generate_competitive_sft.py \
    --input-jsonl data_processed/medec_paired/train_val_split/sft_train.jsonl \
    --output-dir data/sft_competitive_test \
    --prompt-file configs/prompts/gpt4o_competitive_selfplay.json \
    --num-pairs 10 \
    --openai-api-key $OPENAI_API_KEY
```

### Production Run (Competitive)
```bash
python scripts/generate_competitive_sft.py \
    --input-jsonl data_processed/medec_paired/train_val_split/sft_train.jsonl \
    --output-dir data/sft_competitive_production \
    --prompt-file configs/prompts/gpt4o_competitive_selfplay.json \
    --num-pairs 500 \
    --openai-api-key $OPENAI_API_KEY
```

---

## Error Variation Examples

### Medication Errors

**MEDEC Pattern:**
```
lisinopril → losartan (ACE-i → ARB without indication)
```

**Competitive Variations:**
```
- enalapril → valsartan
- ramipril → candesartan
- benazepril → irbesartan
- captopril → olmesartan
```

All maintain pattern: Same error type (ACE-i → ARB), different drug pairs.

### Diagnosis Errors

**MEDEC Pattern:**
```
Huntington disease → Creutzfeldt-Jakob disease (movement disorder confusion)
```

**Competitive Variations:**
```
- Huntington → Progressive supranuclear palsy
- Alzheimer disease → Frontotemporal dementia
- Parkinson disease → Multiple system atrophy
```

All maintain pattern: Similar neurodegenerative conditions requiring differentiation.

### Laboratory Errors

**MEDEC Pattern:**
```
K+ 3.5 → K+ 5.3 (digit transposition)
```

**Competitive Variations:**
```
- Na+ 135 → Na+ 153
- Glucose 120 → 210
- Creatinine 1.2 → 2.1
- WBC 8.5 → 5.8
```

All maintain pattern: Clinically significant digit transposition.

---

## Decision Matrix

Use **Safe Augmentation** if:
- [ ] You only need ~100-200 examples for quick SFT
- [ ] Medical safety is more important than performance
- [ ] You're not planning RL self-play training
- [ ] You want simplest possible approach

Use **Competitive Self-Play** if:
- [x] You're doing full SFT → RL pipeline
- [x] You need 500+ examples
- [x] You want model to generalize, not memorize
- [x] You want competitive self-play training

---

## Summary

**Safe Augmentation:** MEDEC errors as-is, educational reasoning, safest but risks memorization.

**Competitive Self-Play:** Error variations with heavy paraphrasing, prevents memorization, keeps self-play adversarial.

**For MedSeRL with RL training:** Use **Competitive Self-Play** to ensure long-term success of adversarial self-play.
