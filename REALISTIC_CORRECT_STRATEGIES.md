# 5 Realistic Strategies for CORRECT Note Paraphrasing

## Design Philosophy

**Problem:** "Interesting" is too vague - what makes a change realistic but safe?

**Solution:** Base strategies on **real clinical documentation variations** that:
1. Happen in actual medical practice
2. Preserve medical accuracy
3. Require model reasoning to verify safety
4. Are localized (single change only)

---

## The 5 Strategies

### Strategy 1: TEMPORAL EQUIVALENCE
**What it is:** Same duration, different time units

**Real-world scenario:** Different providers express time differently
- Resident: "3 days ago"
- Attending: "72 hours prior to presentation"

**Examples:**
```
✅ "3 days ago" → "72 hours prior to presentation"
✅ "for 1 week" → "over a 7-day period"
✅ "started 2 months ago" → "began 8 weeks ago"
✅ "1-year history" → "12-month history"
```

**Why safe:** Mathematical equivalence (3 days = 72 hours)

**Why realistic:** Different documentation styles across providers

**Model must verify:** Temporal math is correct (not changing duration)

---

### Strategy 2: CLINICAL TERMINOLOGY PRECISION
**What it is:** Lay ↔ Medical term for SAME condition/symptom

**Real-world scenario:** Academic vs community hospital documentation
- Community: "hearing voices"
- Academic: "auditory hallucinations"

**Examples:**
```
✅ "hearing voices" → "auditory hallucinations"
✅ "shortness of breath" → "dyspnea"
✅ "high blood sugar" → "hyperglycemia"
✅ "confused" → "altered mental status"
✅ "committed suicide" → "died by suicide" (preferred phrasing)
✅ "heart attack" → "myocardial infarction"
```

**Why safe:** Same symptom/condition, different terminology level

**Why realistic:** Teaching hospitals use more precise medical terminology

**Model must verify:** Terms refer to SAME condition (not similar conditions)

---

### Strategy 3: ANATOMICAL/LATERALITY DETAIL
**What it is:** Add specific anatomical context to findings

**Real-world scenario:** Attending adds detail during rounds
- Resident: "blood pressure 140/90"
- Attending: "brachial blood pressure 140/90 mmHg"

**Examples:**
```
✅ "blood pressure 140/90" → "brachial blood pressure 140/90 mmHg"
✅ "lung crackles" → "bilateral basilar lung crackles"
✅ "abdominal pain" → "periumbilical abdominal pain"
✅ "weakness" → "right-sided weakness"
```

**Why safe:** Added detail doesn't contradict (if consistent with context)

**Why realistic:** More experienced providers add anatomical precision

**Model must verify:** Added detail is consistent with note context

**CRITICAL:** Only add detail that's already implied or consistent (don't invent laterality that contradicts other findings)

---

### Strategy 4: MEASUREMENT UNIT CONVERSION
**What it is:** Same value, different standard units

**Real-world scenario:** US vs international units, different EMR systems
- Hospital A EMR: "Weight 70 kg"
- Hospital B EMR: "Weight 154 lbs"

**Examples:**
```
✅ "Weight 70 kg" → "Weight 154 lbs" (1 kg = 2.2 lbs)
✅ "Temp 37.8°C" → "Temp 100.0°F" (conversion: C×9/5+32)
✅ "Height 175 cm" → "Height 69 inches" (1 inch = 2.54 cm)
```

**Why safe:** Mathematical conversion preserves value

**Why realistic:** Different hospitals use different unit systems

**Model must verify:** Conversion math is correct

**CRITICAL:** Must use standard units (kg↔lbs, °C↔°F, cm↔inches)

---

### Strategy 5: MEDICATION INDICATION CONTEXT
**What it is:** Add implied medication indication

**Real-world scenario:** Teaching vs community documentation
- Resident: "metformin 1000mg BID"
- Attending: "metformin 1000mg BID for diabetes management"

**Examples:**
```
✅ "metformin 1000mg BID" → "metformin 1000mg BID for diabetes management"
✅ "lisinopril 10mg daily" → "lisinopril 10mg daily for hypertension"
✅ "aspirin 81mg" → "aspirin 81mg for cardiovascular prophylaxis"
```

**Why safe:** Indication is already implied by diagnosis in note

**Why realistic:** Teaching hospitals often state indication explicitly

**Model must verify:** Added indication matches diagnosis in note

**CRITICAL:** Only add indication that's already stated elsewhere in note (don't introduce new conditions)

---

## Application Rules

### ✅ CORRECT Usage

1. **Pick EXACTLY ONE strategy** per note
2. **Apply to EXACTLY ONE location** (one term, one value, one phrase)
3. **Keep everything else IDENTICAL** to original
4. **Verify safety** before finalizing

### ❌ WRONG Usage

- ❌ Multiple strategies in one note
- ❌ Multiple applications of same strategy
- ❌ Trivial changes ("presents" → "visits")
- ❌ Changes that alter medical facts

---

## Examples from Real Notes

### Example 1: Huntington Case

**Original:**
```
...has had multiple episodes of hearing voices...
```

**Strategy 2 Applied:**
```
...has had multiple episodes of auditory hallucinations...
```

**Why safe:** Same symptom (auditory hallucinations = hearing voices)

**Model reasoning required:** Verify both terms describe same phenomenon

---

### Example 2: Diabetes Case

**Original:**
```
She has a 2-year history of depression.
```

**Strategy 1 Applied:**
```
She has a 24-month history of depression.
```

**Why safe:** 2 years = 24 months (temporal equivalence)

**Model reasoning required:** Verify time conversion is mathematically correct

---

### Example 3: Vital Signs

**Original:**
```
Vital signs are within normal limits. Temperature is 37.8 C
```

**Strategy 4 Applied:**
```
Vital signs are within normal limits. Temperature is 100.0 F
```

**Why safe:** 37.8°C = 100.0°F (unit conversion)

**Model reasoning required:** Verify conversion formula applied correctly

---

## Why These Strategies Work

### For SFT Training:
1. **Realistic variations** the model will see in practice
2. **Require reasoning** to verify medical equivalence
3. **Not trivial** - can't just pattern-match synonyms
4. **Verifiable** - localized change, easy to check correctness

### For RL Training:
1. **Foundation patterns** model can build on
2. **Generalizable** - model learns strategy types, not specific instances
3. **Prepares for adversarial play** - model learns to verify carefully

### For Medical Accuracy:
1. **Safe by design** - preserves facts, only changes presentation
2. **Common in practice** - trains model for real-world variation
3. **Easy to verify** - single change, clear equivalence check

---

## Summary

| Strategy | What Changes | Why Safe | Model Must Verify |
|----------|--------------|----------|-------------------|
| 1. Temporal Equivalence | Time units | Math equivalent | Conversion correct |
| 2. Clinical Terminology | Lay↔Medical | Same condition | Terms match |
| 3. Anatomical Detail | Add location | Context-consistent | No contradiction |
| 4. Unit Conversion | Measurement units | Math conversion | Formula correct |
| 5. Medication Context | Add indication | Already implied | Matches diagnosis |

**Key Principle:** All changes preserve medical facts while mimicking real clinical documentation variations.

**Goal:** Train model to VERIFY medical equivalence, not just detect obvious synonyms.
