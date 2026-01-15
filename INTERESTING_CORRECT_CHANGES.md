# Making CORRECT Changes Interesting

## The Problem

**Trivial changes are too easy:**
```
❌ "lives" → "resides"
❌ "note" → "observe"
❌ "Suspected of X" → "X is suspected"
```

**Result:** Model doesn't learn to verify facts, just pattern-matches obvious synonyms.

## The Solution

**Interesting but safe changes require verification:**
```
✅ Reorder: "Diabetes, on metformin 1000mg BID" → "On metformin 1000mg BID for diabetes"
✅ Add context: "metformin 1000mg BID" → "metformin 1000mg BID for glycemic control"
✅ Medical/lay swap: "shortness of breath" → "dyspnea"
✅ Format change: "BP 140/90" → "Blood pressure measured at 140/90 mmHg"
✅ Temporal rephrase: "3 days ago" → "72 hours prior to presentation"
✅ Unit convert: "Temp 37.8°C" → "Temperature 100.04°F"
```

**Result:** Model must THINK to verify medical equivalence!

---

## Safe Paraphrase Strategies

### 1. Reorder Information
**Same facts, different sequence**

❌ **Too obvious:**
```
Original: "Patient has diabetes."
Change: "Patient presents with diabetes."
```

✅ **Interesting:**
```
Original: "Patient has type 2 diabetes controlled on metformin 1000mg BID. HbA1c is 7.2%."
Change: "Patient on metformin 1000mg BID for type 2 diabetes management. Most recent HbA1c measured at 7.2%."
```

**Why interesting:** Diagnosis moved after medication, model must verify all facts present.

---

### 2. Add Redundant Context
**Implied information made explicit**

❌ **Too obvious:**
```
Original: "Patient on lisinopril."
Change: "Patient taking lisinopril."
```

✅ **Interesting:**
```
Original: "Patient on lisinopril 10mg daily."
Change: "Patient on lisinopril 10mg daily for blood pressure management."
```

**Why interesting:** Added indication (implied but not stated), model must verify this doesn't change facts.

---

### 3. Medical/Lay Term Swap
**Same symptom, different terminology**

❌ **Too obvious:**
```
Original: "Patient reports fever."
Change: "Patient states fever."
```

✅ **Interesting:**
```
Original: "Patient reports shortness of breath on exertion."
Change: "Patient reports exertional dyspnea."
```

**Why interesting:** Medical term looks more serious, model must verify same symptom.

**More examples:**
- "high blood sugar" ↔ "hyperglycemia"
- "heart attack" ↔ "myocardial infarction"
- "kidney failure" ↔ "renal insufficiency"
- "stroke" ↔ "cerebrovascular accident"

---

### 4. Format Change
**Same data, different presentation style**

❌ **Too obvious:**
```
Original: "BP 140/90"
Change: "BP: 140/90"
```

✅ **Interesting:**
```
Original: "BP 140/90, HR 88, RR 18"
Change: "Vital signs notable for blood pressure 140/90 mmHg, heart rate 88 beats per minute, and respiratory rate 18 breaths per minute"
```

**Why interesting:** Telegraphic → narrative, model must verify values preserved.

---

### 5. Temporal Rephrasing
**Same timeline, different expression**

❌ **Too obvious:**
```
Original: "Symptoms started 3 days ago."
Change: "Symptoms began 3 days ago."
```

✅ **Interesting:**
```
Original: "Symptoms started 3 days ago."
Change: "Symptoms onset occurred 72 hours prior to presentation."
```

**Why interesting:** Different units (days→hours), model must verify same duration.

**More examples:**
- "for 1 week" → "over a 7-day period"
- "started yesterday" → "initiated approximately 24 hours ago"
- "last month" → "approximately 4 weeks prior"

---

### 6. Unit Conversion
**Same value, different standard units**

❌ **Too obvious (not standard units):**
```
Original: "Weight 70 kg"
Change: "Weight 70000 g"  ← grams not standard for body weight
```

✅ **Interesting (both standard):**
```
Original: "Weight 70 kg"
Change: "Weight 154 lbs"
```

✅ **Interesting (temperature):**
```
Original: "Temperature 37.8°C"
Change: "Temperature 100.04°F"
```

**Why interesting:** Model must verify conversion is correct and both units are medically standard.

**Standard conversions:**
- Weight: kg ↔ lbs
- Temperature: °C ↔ °F
- Height: cm ↔ inches
- ⚠️ Avoid: Non-standard units (grams for weight, etc.)

---

## Examples from Real Notes

### Example 1: Huntington Disease Case

❌ **Trivial change:**
```
Original: "Suspected of Huntington disease"
Change: "Huntington disease is suspected"
```

✅ **Interesting change:**
```
Original: "She has a 2-year history of depression. She was let go by her employer 6 months ago..."
Change: "She has a 2-year history of major depressive disorder. Employment was terminated 6 months ago due to cognitive difficulties..."
```

**Why interesting:**
- "depression" → "major depressive disorder" (medical term)
- "let go" → "employment was terminated" (formal phrasing)
- Added "due to cognitive difficulties" (implied by context)

**Model must verify:** Same medical facts, just more formal/clinical wording.

---

### Example 2: Diabetes with Labs

❌ **Trivial change:**
```
Original: "Patient has type 2 diabetes"
Change: "Patient presents with type 2 diabetes"
```

✅ **Interesting change:**
```
Original: "Patient on metformin 1000mg BID. HbA1c is 7.2%."
Change: "Current glycemic management includes metformin 1000mg twice daily. Hemoglobin A1c level most recently measured at 7.2%."
```

**Why interesting:**
- Restructured presentation (management → medication)
- "BID" → "twice daily" (expansion)
- "HbA1c" → "Hemoglobin A1c level" (full term)
- "is" → "most recently measured at" (more specific)

**Model must verify:** All values and frequencies preserved exactly.

---

### Example 3: Vital Signs

❌ **Trivial change:**
```
Original: "Temperature is 37.8 C"
Change: "Temperature: 37.8 C"
```

✅ **Interesting change:**
```
Original: "Vital signs are within normal limits. Temperature is 37.8 C (100.0 F)"
Change: "Hemodynamically stable with temperature of 100.0°F (37.8°C)"
```

**Why interesting:**
- "Within normal limits" → "Hemodynamically stable" (clinical term)
- Flipped unit order (C first → F first)
- Different presentation style

**Model must verify:** Same vitals, just clinical phrasing + unit reordering.

---

## Benefits

### For Model Training

1. **Requires reasoning:** Can't just pattern-match synonyms
2. **Teaches verification:** Must check medical equivalence
3. **Builds robustness:** Handles different documentation styles
4. **Prepares for RL:** Understands what's safe vs deceptive

### For Self-Play

**CORRECT examples teach:** "These look different but are medically equivalent"
**INCORRECT examples teach:** "These look similar but have subtle errors"

**Together:** Model learns to VERIFY facts, not just match patterns!

---

## Testing

```bash
python scripts/generate_sft_data_safe.py \
    --input-jsonl data_processed/medec_paired/train_val_split/sft_train.jsonl \
    --output-dir data/sft_interesting_test \
    --prompt-file configs/prompts/gpt4o_medec_safe_augmentation.json \
    --num-pairs 3 \
    --add-reasoning \
    --openai-api-key $OPENAI_API_KEY
```

**Expected:** CORRECT changes should look non-trivial but preserve all medical facts!

---

## Summary

❌ **Avoid:** Trivial synonyms ("lives" → "resides", "note" → "observe")

✅ **Use:** Interesting transformations that require verification:
- Reorder information
- Add redundant context
- Swap medical/lay terms
- Change format
- Rephrase temporally
- Convert units

**Goal:** Make CORRECT examples actually test the model's ability to verify medical equivalence!
