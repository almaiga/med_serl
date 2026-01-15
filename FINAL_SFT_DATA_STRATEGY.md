# Final SFT Data Generation Strategy for MedSeRL

## Core Architecture: Dual-Role Training

### The Model Will Play TWO Roles:

1. **INJECTOR**: Generate plausible medical errors
2. **ASSESSOR**: Detect errors in clinical notes

### Training Data Structure: 50/50 Split

- **50% CORRECT** notes (safe paraphrases from MEDEC)
- **50% INCORRECT** notes (MEDEC errors AS-IS)

---

## Data Source: MEDEC Dataset

**What is MEDEC?**
- Doctor-verified medical error pairs
- Each pair has: `correct_note` + `incorrect_note`
- Errors are realistic (from actual clinical practice)
- Safe to use (already verified by physicians)

**Why MEDEC?**
- ✅ Doctor-verified errors (no risk of teaching wrong patterns)
- ✅ Realistic error types (diagnosis, medication, lab values)
- ✅ Paired data (perfect for self-play training)

---

## CORRECT Examples: 5 Realistic Strategies

**Goal:** Teach model to verify medical equivalence through safe transformations

### Strategy 1: TEMPORAL EQUIVALENCE
- Change: "3 days ago" → "72 hours prior to presentation"
- Safe because: Mathematical equivalence (3 days = 72 hours)
- Model learns: Verify time conversion is correct

### Strategy 2: CLINICAL TERMINOLOGY PRECISION
- Change: "hearing voices" → "auditory hallucinations"
- Safe because: Same symptom, different terminology level
- Model learns: Verify terms refer to SAME condition

### Strategy 3: ANATOMICAL/LATERALITY DETAIL
- Change: "blood pressure 140/90" → "brachial blood pressure 140/90 mmHg"
- Safe because: Added detail consistent with context
- Model learns: Verify added detail doesn't contradict

### Strategy 4: MEASUREMENT UNIT CONVERSION
- Change: "Weight 70 kg" → "Weight 154 lbs"
- Safe because: Mathematical conversion (1 kg = 2.2 lbs)
- Model learns: Verify conversion formula is correct

### Strategy 5: MEDICATION INDICATION CONTEXT
- Change: "metformin 1000mg BID" → "metformin 1000mg BID for diabetes"
- Safe because: Indication already implied in note
- Model learns: Verify added context matches diagnosis

**CRITICAL RULE:** Make EXACTLY ONE change per note (localized, verifiable)

---

## INCORRECT Examples: MEDEC AS-IS

**Goal:** Teach model realistic error injection strategies

**Source:** Use MEDEC `incorrect_note` directly (no modification)

**Why safe:** Doctor-verified errors, no risk of teaching wrong patterns

**Example error types:**
- Diagnosis confusion (Huntington → CJD)
- Medication errors (azithromycin → erythromycin)
- Lab value errors (osteomalacia → osteoporosis)

---

## Chain-of-Thought Reasoning (NEW!)

### For CORRECT Notes (Verification Chain)

**Purpose:** Teach model HOW to verify safety step-by-step

**Format:**
```json
{
  "change_made": "Changed 'hearing voices' → 'auditory hallucinations'",
  "verification_chain": [
    "Step 1: Only one term changed - rest identical",
    "Step 2: Check medical equivalence - both describe same symptom",
    "Step 3: Verify facts preserved - age, timeline, other symptoms unchanged",
    "Step 4: Strategy 2 applied correctly - lay/medical term swap"
  ],
  "why_safe": "Both terms describe same symptom, all facts preserved",
  "strategy_used": "Strategy 2: CLINICAL TERMINOLOGY PRECISION",
  "preserved_facts": ["Age: 42", "Timeline: 1-year history", "Other symptoms unchanged"]
}
```

**What model learns:**
1. HOW to verify medical equivalence (not just "it's correct")
2. WHICH strategy was used (generalizable pattern)
3. WHAT to check (step-by-step verification)

### For INCORRECT Notes (Deception Chain)

**Purpose:** Teach model HOW to inject errors strategically

**Format:**
```json
{
  "injection_strategy": "Changed 'Huntington disease' → 'Creutzfeldt-Jakob disease'",
  "deception_chain": [
    "Step 1: Found similar disease - both neurodegenerative",
    "Step 2: Verified symptom overlap - movements, cognitive decline",
    "Step 3: Located differentiator - family history (genetic vs sporadic)",
    "Step 4: Bet assessor misses it - history buried in note"
  ],
  "why_plausible": "Both rare with overlapping symptoms",
  "detection_requires": "Must check family history and timeline carefully",
  "detection_clues": [
    "Father suicide at 50 - genetic Huntington",
    "1-year progression - too slow for CJD",
    "Choreiform movements - typical Huntington"
  ],
  "error_type_pattern": "Similar disease presentations"
}
```

**What model learns:**
1. HOW to find plausible alternatives (injector strategy)
2. WHY it's deceptive (what assessor might miss)
3. WHAT to check (detection strategy for assessor)

---

## Training Flow

### Phase 1: SFT Training

**Input:** 1000 examples (500 CORRECT + 500 INCORRECT)

**Model learns TWO tasks:**

#### Task A: INJECTOR Role
```
Input: Original clinical note
Output: Modified note with reasoning

For CORRECT:
- Apply ONE of 5 strategies
- Output: safe paraphrase + verification chain

For INCORRECT:
- Learn from MEDEC error patterns
- Output: error + deception chain
```

#### Task B: ASSESSOR Role
```
Input: Clinical note (CORRECT or INCORRECT)
Output: Classification + reasoning

Model must:
- Analyze note systematically
- Check for errors using detection clues
- Output: CORRECT/INCORRECT + explanation
```

**Prompt Format for SFT:**
```
You are a medical AI that can play two roles:

ROLE 1 - INJECTOR: Generate plausible medical errors
<think>
[Deception chain: how to make error plausible]
</think>
<output>[Modified note with error]</output>

ROLE 2 - ASSESSOR: Detect errors in clinical notes
<think>
[Verification chain: check for errors systematically]
</think>
<answer>CORRECT or INCORRECT</answer>
```

### Phase 2: RL Self-Play Training

**After SFT, model has learned:**
- ✅ 5 safe transformation strategies (CORRECT)
- ✅ Error injection patterns from MEDEC (INCORRECT)
- ✅ Verification reasoning (assessor)
- ✅ Deception reasoning (injector)

**During RL:**
1. Sample batch of notes (32/64/128)
2. **Injector generates** modifications (frozen, then policy update)
3. **VCF filters** (Jaccard, word edits, single error check)
4. **Assessor evaluates** (CORRECT/INCORRECT)
5. **Compute rewards** (zero-sum: injector vs assessor)
6. **Policy update** (REINFORCE++)
7. Repeat with new batch

**Model becomes creative:**
- Injector explores variations of learned strategies
- Assessor learns to catch novel errors
- Self-play drives adversarial improvement

---

## Benefits of Chain-of-Thought Reasoning

### For SFT:
1. **Teaches verification process** - Not just "it's correct" but WHY
2. **Strategy awareness** - Model knows WHICH pattern it's using
3. **Generalizable learning** - Learns reasoning, not just examples

### For RL:
1. **Better rollout reasoning** - Model thinks step-by-step during generation
2. **Creative exploration** - Understanding strategy helps generate variations
3. **Strategic deception** - Injector learns HOW to fool assessor
4. **Systematic detection** - Assessor learns WHAT to check

### For Medical Safety:
1. **Explainable errors** - Can trace WHY error is plausible
2. **Verifiable safety** - Clear reasoning for CORRECT changes
3. **Training transparency** - Reasoning shows model's thought process

---

## Data Generation Command

### Test (2 pairs):
```bash
python3 scripts/generate_sft_data_safe.py \
    --input-jsonl data_processed/medec_paired/train_val_split/sft_train.jsonl \
    --output-dir data/sft_cot_test \
    --prompt-file configs/prompts/gpt4o_medec_safe_augmentation.json \
    --num-pairs 2 \
    --add-reasoning \
    --openai-api-key "$OPENAI_API_KEY"
```

### Production (500 pairs = 1000 examples):
```bash
python3 scripts/generate_sft_data_safe.py \
    --input-jsonl data_processed/medec_paired/train_val_split/sft_train.jsonl \
    --output-dir data/sft_production \
    --prompt-file configs/prompts/gpt4o_medec_safe_augmentation.json \
    --num-pairs 500 \
    --add-reasoning \
    --openai-api-key "$OPENAI_API_KEY"
```

**Expected:**
- Output: 1000 examples (500 CORRECT + 500 INCORRECT)
- Cost: ~$20-30 (with chain-of-thought reasoning)
- Time: 1-2 hours with rate limiting

---

## Output Format

### CORRECT Example:
```json
{
  "note": "[Modified clinical note with ONE safe change]",
  "label": "CORRECT",
  "original_note_id": "ms-train-418",
  "source": "medec_correct_paraphrased",
  "metadata": {
    "generation_type": "correct_paraphrase",
    "gpt4o_tokens": 1500
  },
  "reasoning": {
    "change_made": "Changed 'hearing voices' → 'auditory hallucinations'",
    "verification_chain": [
      "Step 1: Only one term changed",
      "Step 2: Medical equivalence verified",
      "Step 3: All facts preserved",
      "Step 4: Strategy 2 applied correctly"
    ],
    "why_safe": "Both terms describe same symptom",
    "strategy_used": "Strategy 2: CLINICAL TERMINOLOGY PRECISION",
    "preserved_facts": ["Age", "Timeline", "Other symptoms"]
  }
}
```

### INCORRECT Example:
```json
{
  "note": "[MEDEC incorrect_note AS-IS]",
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
    "injection_strategy": "Changed 'Huntington' → 'CJD'",
    "deception_chain": [
      "Step 1: Found similar disease",
      "Step 2: Verified symptom overlap",
      "Step 3: Located differentiator",
      "Step 4: Bet assessor misses it"
    ],
    "why_plausible": "Both rare neurodegenerative diseases",
    "detection_requires": "Check family history and timeline",
    "detection_clues": ["Father suicide at 50", "1-year progression"],
    "error_type_pattern": "Similar disease presentations"
  }
}
```

---

## Quality Checks

### For CORRECT:
- [ ] Only ONE change made (verify with diff)
- [ ] Change uses ONE of 5 strategies
- [ ] All medical facts preserved
- [ ] Verification chain has 4 steps
- [ ] Strategy explicitly labeled

### For INCORRECT:
- [ ] Identical to MEDEC incorrect_note
- [ ] Deception chain has 4 steps
- [ ] Detection clues provided
- [ ] Error pattern identified

---

## Summary: Why This Works

### SFT Phase:
1. **Teaches task format** - Model learns injector/assessor roles
2. **Provides reasoning** - Chain-of-thought shows HOW to verify/inject
3. **Strategy awareness** - Model learns 5 CORRECT patterns + MEDEC error patterns
4. **Balanced data** - 50/50 prevents bias

### RL Phase:
1. **Creative exploration** - Model generates variations of learned strategies
2. **Adversarial improvement** - Zero-sum rewards drive competitive play
3. **Strategic reasoning** - Model thinks step-by-step during rollouts
4. **Generalizable patterns** - Learns reasoning process, not just examples

### Medical Safety:
1. **Verified errors** - MEDEC errors are doctor-approved
2. **Safe transformations** - 5 strategies preserve medical accuracy
3. **Explainable reasoning** - Chain-of-thought shows model's logic
4. **Localized changes** - Single modifications are easy to verify

**Key Principle:** SFT teaches format + reasoning, RL drives creativity + deception!
