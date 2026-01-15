# MEDEC Dataset Analysis & Prompt Optimization

## MEDEC Note Characteristics

Based on analysis of `sft_train.jsonl`:

### Structure
- **Length**: 200-400 words per vignette
- **Format**: Clinical case presentations (similar to USMLE/board exam style)
- **Content**: Comprehensive patient presentations including:
  - Demographics and chief complaint
  - Detailed history (HPI, PMH, social history)
  - Complete vital signs
  - Physical exam findings (often extensive)
  - Laboratory panels (complete metabolic, CBC, etc.)
  - Diagnosis and management plan

### Error Types in MEDEC (from samples)

**1. Diagnostic Errors (45%)** - Most common:
- Similar disease presentations:
  - `Huntington disease` ↔ `Creutzfeldt-Jakob disease` (both: movement disorders, cognitive decline)
  - `Osteomalacia` ↔ `Osteoporosis` (differentiated by PTH levels)
  - `TTP` ↔ `DIC` (both: thrombocytopenia + anemia, different fibrin products)
  - `Aldosteronoma` ↔ `Pheochromocytoma` (both: hypertension + metabolic, different K+ levels)

**2. Management/Treatment Errors (30%)**:
- Wrong drug selection (same indication, different choice):
  - `Azithromycin` ↔ `Erythromycin` (both macrolides, azithro preferred for pertussis)
  - `Salmeterol` ↔ `Fluticasone` (asthma control - LABA vs ICS)
- Wrong procedure/approach:
  - `Percutaneous surgery` ↔ `Indomethacin infusion` (PDA management)
  - `Closed reduction` ↔ `Test sensation` (shoulder dislocation - check axillary nerve first)
- Wrong fluids/concentrations:
  - `0.9% saline` ↔ `0.45% saline` (isotonic vs hypotonic for dehydration)

**3. Treatment Errors (20%)**:
- Medication frequency: `BID` ↔ `TID` (dosing errors)
- Wrong diagnostic workup order

**4. Laboratory Interpretation (5%)**:
- Misdiagnosis despite clear lab findings

### Key Subtlety Patterns

**What makes MEDEC errors subtle:**

1. **Clinical plausibility**: Both conditions fit the presentation
   - Example: Huntington vs CJD - both cause movement + cognitive symptoms
   - The *family history of suicide at 50* is the key differentiator (genetic Huntington)

2. **Same category, different choice**: Similar drugs/procedures
   - Example: Erythromycin vs Azithromycin - both work, but azithro is first-line
   - Example: Salmeterol vs Fluticasone - both for asthma, but need ICS before LABA

3. **Lab context matters**: Diagnosis requires interpreting full panel
   - Example: Osteoporosis vs Osteomalacia - PTH 855 pg/mL points to osteomalacia
   - Example: Aldosteronoma vs Pheochromocytoma - K+ 3.0 suggests aldosteronoma

4. **Guideline violations**: Breaks standard of care
   - Example: LABA monotherapy in asthma (should start ICS first)
   - Example: Closed reduction before neurologic check (should assess axillary nerve)

## Prompt Optimizations

Created `configs/prompts/gpt4o_medec_optimized.json` with:

### 1. Error Categories Matched to MEDEC
```json
{
  "diagnosis": 0.45,      // Similar presentations
  "management": 0.30,     // Wrong approach/procedure
  "treatment": 0.20,      // Wrong medication selection
  "laboratory": 0.05      // Misinterpretation
}
```

### 2. VCF Thresholds for Long Vignettes
```json
{
  "correct": {
    "min_jaccard": 0.70,  // Allow 30% word changes
    "max_jaccard": 0.98   // Substantial paraphrasing OK
  },
  "incorrect": {
    "min_jaccard": 0.85,  // 85-99% similarity
    "max_jaccard": 0.99,  // Subtle single errors
    "max_word_edits": 50  // ~5-50 words changed in 300+ word vignettes
  }
}
```

### 3. Specific Error Examples from MEDEC
Included in prompts:
- Huntington ↔ CJD (diagnostic)
- Azithromycin ↔ Erythromycin (management)
- 0.9% ↔ 0.45% saline (treatment)
- Osteomalacia ↔ Osteoporosis (lab interpretation)

### 4. Clinical Context Emphasis
Prompts emphasize:
- Understanding full clinical picture before injecting error
- Choosing plausible alternatives (not absurd ones)
- Maintaining natural medical writing
- Single error only (no cascading mistakes)

## Example from Dataset

### Original (CORRECT):
```
A 42-year-old woman... Suspected of Huntington disease. Physical examination
shows irregular, nonrepetitive, and arrhythmic movements...
```

### With Error (INCORRECT):
```
A 42-year-old woman... Suspected of Creutzfeldt-Jakob disease. Physical
examination shows irregular, nonrepetitive, and arrhythmic movements...
```

**Why this error is subtle:**
- Both cause movement disorders + cognitive decline
- Both have psychiatric symptoms
- Both have similar neurologic exam findings
- **KEY DIFFERENTIATOR**: Family history of suicide at 50 (Huntington is genetic, typically manifests 30s-50s)

## Generation Strategy

### For SFT Training Data:

**Phase 1: Generate from MEDEC**
```bash
python scripts/generate_sft_data_advanced.py \
    --input-jsonl data_processed/medec_paired/train_val_split/sft_train.jsonl \
    --output-dir data/sft_medec_gpt4o \
    --prompt-file configs/prompts/gpt4o_medec_optimized.json \
    --num-correct 500 \
    --num-incorrect 500 \
    --openai-api-key $OPENAI_API_KEY
```

**Phase 2: SFT Train Qwen3-4B**
```bash
# Train Qwen3-4B on GPT-4o generated data
python train_sft.py \
    --train-data data/sft_medec_gpt4o/sft_combined_advanced.jsonl \
    --model Qwen/Qwen3-4B \
    --epochs 3
```

**Phase 3: RL Self-Play**
```bash
# Now Qwen3-4B can generate errors for RL training
bash scripts/train_medserl_reinforce_pp.sh
```

## Quality Checks

### For CORRECT paraphrases:
✅ All numerical values preserved (labs, vitals, doses)
✅ All diagnoses preserved (can use synonyms)
✅ Clinical logic maintained
✅ Natural medical writing
✅ Jaccard 0.70-0.98 (substantial variation)

### For INCORRECT errors:
✅ Error type matches MEDEC categories
✅ Error is medically plausible
✅ Only ONE error introduced
✅ Natural writing (no markers)
✅ Jaccard 0.85-0.99 (subtle change)
✅ Requires medical knowledge to detect

## Cost Estimate

**For 1000 examples (500 CORRECT + 500 INCORRECT):**
- Average MEDEC vignette: ~300 words = ~400 tokens
- Generation: ~600 tokens per example
- Verification (CORRECT only): ~800 tokens per example

**Total tokens:**
- CORRECT: 500 × (400 input + 600 output + 800 verification) = 900K tokens
- INCORRECT: 500 × (400 input + 600 output) = 500K tokens
- **Total: ~1.4M tokens**

**Cost estimate:**
- Input: 1.4M × $2.50/1M = ~$3.50
- Output: 1.4M × $10/1M = ~$14
- **Total: ~$17-20 for 1000 examples**

## Next Steps

1. ✅ **Done**: Analyzed MEDEC structure and error patterns
2. ✅ **Done**: Created optimized prompts (`gpt4o_medec_optimized.json`)
3. ✅ **Done**: Calibrated VCF thresholds for long vignettes
4. ✅ **Done**: Mapped error type distributions

**Ready to generate:**
```bash
# Test with small batch first
python scripts/generate_sft_data_advanced.py \
    --input-jsonl data_processed/medec_paired/train_val_split/sft_train.jsonl \
    --output-dir data/sft_test \
    --prompt-file configs/prompts/gpt4o_medec_optimized.json \
    --num-correct 25 \
    --num-incorrect 25 \
    --openai-api-key $OPENAI_API_KEY

# Review quality, then scale to 500-1000 examples
```

## Summary

The MEDEC dataset contains sophisticated medical vignettes with subtle errors that require deep clinical reasoning to detect. The prompts are optimized to:

1. **Generate realistic errors** matching MEDEC patterns
2. **Maintain subtlety** (85-99% similarity, single errors)
3. **Preserve clinical plausibility** (similar conditions, not absurd alternatives)
4. **Support diverse error types** (diagnosis, management, treatment, labs)

These prompts will create high-quality SFT training data to teach Qwen3-4B the task before RL self-play training.
