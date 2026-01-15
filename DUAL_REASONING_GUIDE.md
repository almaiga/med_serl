# Dual Reasoning: Teaching Task Understanding

## The Goal

Teach the model to understand **TWO complementary skills**:
1. **Safe paraphrasing** (CORRECT) - How to modify without introducing errors
2. **Strategic error injection** (INCORRECT) - How to make plausible but wrong changes

## Why Both Matter

### CORRECT Reasoning (Safe Changes)
```json
{
  "note": "Patient presents with type 2 diabetes...",
  "label": "CORRECT",
  "reasoning": {
    "change_made": "Changed 'has' to 'presents with'",
    "why_safe": "Only verb changed, diagnosis preserved",
    "preserved_facts": ["Diagnosis unchanged", "All meds unchanged"],
    "paraphrase_technique": "Synonym substitution",
    "medical_equivalence": "Both phrasings mean the same thing"
  }
}
```

**Model learns:**
- What changes are SAFE (preserve medical meaning)
- How to paraphrase without introducing errors
- What facts are critical (must preserve)
- What can vary (synonyms, style)

### INCORRECT Reasoning (Plausible but Wrong)
```json
{
  "note": "...Suspected of Creutzfeldt-Jakob disease...",
  "label": "INCORRECT",
  "reasoning": {
    "injection_strategy": "Changed Huntington → CJD",
    "why_plausible": "Both cause movement disorders and dementia",
    "deception_technique": "Overlapping symptoms fool assessor",
    "detection_clues": ["Family history", "Timeline"],
    "error_type_pattern": "Similar disease confusion"
  }
}
```

**Model learns:**
- What changes are DECEPTIVE (look plausible but wrong)
- How to inject errors strategically
- What makes errors hard to detect
- General patterns to apply

---

## The Complete Picture

```
CORRECT Example (teaches safe modification):
  Original: "Patient has type 2 diabetes on metformin 1000mg BID."
  Paraphrased: "Patient presents with type 2 diabetes on metformin 1000mg BID."
  Reasoning: "Changed verb (has → presents), diagnosis and meds preserved exactly."

  Model learns: "I can change phrasing but must preserve critical facts"

INCORRECT Example (teaches strategic error):
  Original: "Oral azithromycin is administered..."
  With Error: "Oral erythromycin is administered..."
  Reasoning: "Changed azithromycin → erythromycin (both macrolides, but azithro preferred)"

  Model learns: "I can substitute drugs in same class to create plausible error"
```

---

## Training Data Structure

### CORRECT Example (Full)
```json
{
  "note": "A 42-year-old woman presents to the physician...",
  "label": "CORRECT",
  "original_note_id": "ms-train-418",
  "source": "medec_correct_paraphrased",
  "metadata": {
    "generation_type": "correct_paraphrase",
    "gpt4o_tokens": 715
  },
  "reasoning": {
    "change_made": "Changed 'is brought to' to 'presents to'",
    "why_safe": "Both phrases indicate patient arrival at physician, no clinical difference",
    "preserved_facts": [
      "Patient demographics (42-year-old woman)",
      "Accompanied by husband (unchanged)",
      "All symptoms and timeline preserved",
      "All diagnoses unchanged"
    ],
    "paraphrase_technique": "Synonym substitution for arrival phrase",
    "medical_equivalence": "'Is brought to' and 'presents to' both describe patient-physician encounter initiation"
  }
}
```

### INCORRECT Example (Full)
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

---

## What Model Learns from Dual Reasoning

### Phase 1: Task Format (SFT)

**From CORRECT examples:**
- Input format: Clinical note
- Output format: Modified note
- Safe modifications: Synonym, restructure, style
- Critical facts: Diagnosis, meds, labs, timeline

**From INCORRECT examples:**
- Input format: Clinical note
- Output format: Modified note with subtle error
- Strategic errors: Similar conditions, drug class, digit transposition
- Deception: Overlapping symptoms, plausible alternatives

### Phase 2: Pattern Learning (SFT)

**CORRECT patterns:**
- Verb substitution (has → presents)
- Style change (telegraphic ↔ full sentences)
- Terminology (dyspnea ↔ shortness of breath)
- Structure (reorder without changing meaning)

**INCORRECT patterns:**
- Similar diagnoses (Huntington → CJD, Alzheimer → Frontotemporal dementia)
- Drug class confusion (lisinopril → losartan, azithromycin → erythromycin)
- Digit transposition (3.5 → 5.3, K+ → Na+)
- Laterality errors (right → left for unilateral conditions)

### Phase 3: Generalization (RL)

**Model combines learned patterns:**
- "I learned 'similar diagnosis' pattern from Huntington/CJD"
- "I can apply to Parkinson/MSA (similar movement disorders)"
- "Policy gradient rewards deceptive applications"

---

## Generation Example

### Input (MEDEC pair)
```
correct_note: "...Suspected of Huntington disease..."
incorrect_note: "...Suspected of Creutzfeldt-Jakob disease..."
error_type: "diagnosis"
```

### Output (With dual reasoning)

**CORRECT:**
```json
{
  "note": "[Minimal paraphrase of correct_note]",
  "label": "CORRECT",
  "reasoning": {
    "change_made": "[What was changed]",
    "why_safe": "[Why medical facts preserved]",
    "preserved_facts": ["Fact 1", "Fact 2"],
    "paraphrase_technique": "[Strategy]",
    "medical_equivalence": "[Why same meaning]"
  }
}
```

**INCORRECT:**
```json
{
  "note": "[MEDEC incorrect_note AS-IS]",
  "label": "INCORRECT",
  "reasoning": {
    "injection_strategy": "[What error was made]",
    "why_plausible": "[Why looks realistic]",
    "deception_technique": "[What makes it subtle]",
    "clinical_reasoning": "[Why error occurs]",
    "detection_clues": ["Clue 1", "Clue 2"],
    "error_type_pattern": "[General pattern]"
  }
}
```

---

## Benefits

### 1. Task Understanding
Model learns distinction between:
- Safe modifications (CORRECT)
- Deceptive modifications (INCORRECT)

### 2. Strategy Learning
Model learns HOW, not just WHAT:
- HOW to preserve facts (CORRECT)
- HOW to inject errors (INCORRECT)

### 3. Pattern Generalization
Model learns reusable patterns:
- Paraphrase techniques (CORRECT)
- Error injection strategies (INCORRECT)

### 4. RL Readiness
Model has foundation for:
- Exploring safe variations (CORRECT)
- Generating novel deceptive errors (INCORRECT)

---

## Testing

### Command
```bash
python scripts/generate_sft_data_safe.py \
    --input-jsonl data_processed/medec_paired/train_val_split/sft_train.jsonl \
    --output-dir data/sft_dual_reasoning_test \
    --prompt-file configs/prompts/gpt4o_medec_safe_augmentation.json \
    --num-pairs 3 \
    --add-reasoning \
    --openai-api-key $OPENAI_API_KEY
```

### Expected Output
- 3 CORRECT examples with "safe paraphrase" reasoning
- 3 INCORRECT examples with "injector strategy" reasoning
- Both types teach complementary skills

---

## Summary

**Dual reasoning teaches:**

✅ **CORRECT:** How to modify safely (preserve medical accuracy)
✅ **INCORRECT:** How to inject strategically (create plausible errors)
✅ **Both:** Give model complete task understanding
✅ **Result:** Model becomes effective injector (creative during RL)

**Key insight:** Model needs to understand BOTH sides of the task to be effective!
