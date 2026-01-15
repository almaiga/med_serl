# Injector Reasoning for SFT Training

## The Problem

Without reasoning, the model sees:
```json
{
  "note": "...Suspected of Creutzfeldt-Jakob disease...",
  "label": "INCORRECT"
}
```

**Model learns:** "This note is incorrect" (but not HOW or WHY to create such errors)

## The Solution: Injector Reasoning

With injector reasoning, the model sees:
```json
{
  "note": "...Suspected of Creutzfeldt-Jakob disease...",
  "label": "INCORRECT",
  "reasoning": {
    "injection_strategy": "Changed 'Huntington disease' to 'Creutzfeldt-Jakob disease'",
    "why_plausible": "Both are neurodegenerative diseases with movement disorders and cognitive decline",
    "deception_technique": "Symptoms overlap - without checking family history, they appear similar",
    "clinical_reasoning": "Error occurs because both are rare, both affect movement/cognition",
    "detection_clues": ["Family history genetic pattern", "1-year vs rapid progression"],
    "error_type_pattern": "Similar disease presentations requiring differential diagnosis"
  }
}
```

**Model learns:**
- HOW to inject errors (change diagnosis to similar condition)
- WHY it's plausible (overlapping symptoms)
- WHAT makes it deceptive (need careful analysis to detect)
- GENERAL PATTERN (similar presentations)

---

## Two Types of Reasoning

### ❌ Medical Reasoning (what we DON'T want)
**Perspective:** Assessor/educator explaining why error is wrong
```json
{
  "error_identified": "Wrong diagnosis",
  "why_wrong": "Should be Huntington, not CJD",
  "clinical_impact": "Affects genetic counseling"
}
```
**Teaches:** Medical facts, error detection

### ✅ Injector Reasoning (what we DO want)
**Perspective:** Injector explaining injection strategy
```json
{
  "injection_strategy": "Changed Huntington → CJD",
  "why_plausible": "Both cause movement disorders",
  "deception_technique": "Overlapping symptoms fool assessor",
  "error_type_pattern": "Similar disease confusion"
}
```
**Teaches:** How to generate realistic, deceptive errors

---

## Why This Matters for SFT

### Without Injector Reasoning
```
SFT Training:
  Input: Clinical note
  Output: Note with error
  Model: ???

Result: Model generates random/garbage changes
```

### With Injector Reasoning
```
SFT Training:
  Input: Clinical note
  Output: Note with error + injection strategy
  Model learns:
    - What change was made
    - Why it's plausible
    - How to fool assessor
    - General pattern to apply

Result: Model generates REALISTIC, STRATEGIC errors
```

---

## Reasoning Structure

```json
{
  "injection_strategy": "Specific change made (diagnosis, medication, lab value)",

  "why_plausible": "Why this error looks realistic to human/assessor",

  "deception_technique": "What makes it subtle and hard to detect",

  "clinical_reasoning": "Why this error occurs in real medical practice",

  "detection_clues": [
    "Clue 1 assessor needs to catch it",
    "Clue 2",
    "Clue 3"
  ],

  "error_type_pattern": "General pattern for this error type"
}
```

---

## Example: Diagnosis Error

### MEDEC Pair
**Correct:** "Suspected of Huntington disease"
**Incorrect:** "Suspected of Creutzfeldt-Jakob disease"

### Injector Reasoning (Teaches Strategy)
```json
{
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
```

**What Model Learns:**
1. **Strategy:** Change diagnosis to similar condition
2. **Selection:** Pick conditions with overlapping symptoms
3. **Deception:** Works when key differentiators are subtle
4. **Pattern:** "Similar presentation confusion" generalizes to other cases

---

## Example: Medication Error

### MEDEC Pair
**Correct:** "Oral azithromycin is administered"
**Incorrect:** "Oral erythromycin is administered"

### Injector Reasoning
```json
{
  "injection_strategy": "Changed antibiotic from 'azithromycin' to 'erythromycin'",

  "why_plausible": "Both are macrolide antibiotics used for pertussis, so the substitution looks medically reasonable",

  "deception_technique": "Same drug class makes it hard to spot - assessor must know that azithromycin is PREFERRED due to better tolerability and fewer side effects",

  "clinical_reasoning": "This error occurs when clinicians know both work but don't remember which is first-line, or in formulary restrictions",

  "detection_clues": [
    "Azithromycin is first-line for pertussis (better GI tolerance)",
    "Erythromycin has more side effects (nausea, drug interactions)",
    "With newborn in household, need reliable treatment adherence"
  ],

  "error_type_pattern": "Same drug class, suboptimal choice - requires knowing first-line vs alternative agents"
}
```

**What Model Learns:**
1. **Strategy:** Substitute within same drug class
2. **Selection:** Pick drugs that both work but one is preferred
3. **Deception:** Looks reasonable without guideline knowledge
4. **Pattern:** "Suboptimal drug selection" generalizes to other medication choices

---

## Example: Laboratory Error

### MEDEC Pair
**Correct:** "K+ 3.5 mEq/L"
**Incorrect:** "K+ 5.3 mEq/L"

### Injector Reasoning
```json
{
  "injection_strategy": "Transposed digits in potassium value: 3.5 → 5.3",

  "why_plausible": "Digit transposition is a common transcription error (3.5 vs 5.3 - just swapped)",

  "deception_technique": "Both values are physiologically possible, but 5.3 is hyperkalemia (clinically significant). Assessor must know this changes management urgency.",

  "clinical_reasoning": "This error occurs during manual data entry, copy-paste mistakes, or voice recognition errors",

  "detection_clues": [
    "5.3 mEq/L is hyperkalemia (needs urgent treatment)",
    "3.5 mEq/L is normal-low (may need monitoring)",
    "Clinical context: patient symptoms don't match hyperkalemia"
  ],

  "error_type_pattern": "Digit transposition - common data entry error with clinical significance"
}
```

**What Model Learns:**
1. **Strategy:** Transpose digits in lab values
2. **Selection:** Pick values where swap is plausible but significant
3. **Deception:** Both values possible, but one changes management
4. **Pattern:** "Transcription error" generalizes to other numerical data

---

## Benefits for RL Training

### During SFT (with injector reasoning)
- Model learns **error generation strategies**
- Understands **what makes errors deceptive**
- Learns **general patterns** to apply

### During RL (builds on SFT)
- Model explores **variations** of learned strategies
- Policy gradient pushes toward **more deceptive** errors
- Generalizes patterns to **novel cases**

### Example Progression

**SFT learns:**
- "Change similar diagnoses (Huntington → CJD)"
- "Substitute same-class drugs (azithromycin → erythromycin)"

**RL explores:**
- "Change Alzheimer → Frontotemporal dementia" (same pattern)
- "Substitute lisinopril → losartan" (same pattern, different drugs)

**Result:** Model becomes creative error injector!

---

## Implementation

### Current Status
✅ Prompts updated to generate injector reasoning
✅ Script updated to pass correct parameters
✅ Example output shows injector perspective

### Test Command
```bash
bash test_injector_reasoning.sh
```

### Expected Output
```json
{
  "note": "...Suspected of Creutzfeldt-Jakob disease...",
  "label": "INCORRECT",
  "reasoning": {
    "injection_strategy": "Changed 'Huntington disease' to 'Creutzfeldt-Jakob disease'",
    "why_plausible": "Both neurodegenerative with movement disorders",
    "deception_technique": "Overlapping symptoms fool assessor",
    "clinical_reasoning": "Both rare, both affect movement/cognition",
    "detection_clues": ["Family history pattern", "Timeline mismatch"],
    "error_type_pattern": "Similar disease presentations"
  }
}
```

---

## Key Takeaway

**Injector reasoning teaches the MODEL how to be a strategic error generator, not just a passive classifier.**

This is essential because:
1. SFT without reasoning → garbage errors
2. SFT with injector reasoning → strategic, realistic errors
3. RL builds on learned strategies → creative adversarial injector

**The reasoning is the curriculum that teaches the injector role!**
