# Final SFT Data Generation Strategy

## Core Principle: Simple SFT, Creative RL

**Key Insight:** SFT teaches format, RL drives deception. Keep SFT simple!

---

## The Strategy

### CORRECT Examples
- **Paraphrase ONE sentence only**
- Keep rest of note identical
- Easy to verify correctness

### INCORRECT Examples
- **Use MEDEC AS-IS** (doctor-verified)
- No variations needed
- Optional: Add reasoning (educational)

### Split
- **50% CORRECT** (minimal paraphrase)
- **50% INCORRECT** (MEDEC as-is)

---

## Why This Works

### SFT Phase (Simple Format Learning)
```
Goal: Teach model the task
Data: Minimal paraphrasing + MEDEC errors
Model learns:
  - Input format: Clinical note
  - Output format: Modified note with metadata
  - Error types: What errors look like
  - Task structure: Inject/detect errors
```

### RL Phase (Creative Adversarial Learning)
```
Goal: Generate deceptive errors
Process: Self-play with frozen assessor
Model learns:
  - Sample from learned distribution
  - Generate NEW notes each rollout
  - Explore error variations
  - Fool assessor via policy gradient

Result: Model becomes creative during RL, not SFT!
```

---

## Why Minimal Paraphrasing?

### ✅ Advantages

1. **Easy to verify**: Can quickly check medical facts preserved
2. **Lower risk**: Less chance of introducing errors during generation
3. **Cheaper**: Fewer tokens, faster generation
4. **Sufficient for SFT**: Just need format diversity, not heavy variation
5. **RL handles creativity**: Model will explore during self-play

### ❌ Heavy Paraphrasing Problems

1. **Hard to verify**: 30-50% changes make correctness checking difficult
2. **Higher risk**: More opportunities to change critical medical facts
3. **Unnecessary**: SFT just teaches format, not deception
4. **Expensive**: More GPT-4o tokens for verification

---

## Memorization: Why It's Not a Problem

### During SFT Training
- Model sees ~500-1000 examples
- Learns task format and error patterns
- Some memorization is OK - we're teaching basics

### During RL Training (Where it matters)
- **Model generates fresh data each rollout**
- Policy samples from learned distribution
- Exploration via temperature/sampling
- Each episode has NEW note variations
- Assessor can't memorize (data is generated on-the-fly)

### The Magic of RL Self-Play
```
Round 1:
  Injector: "Patient on lisinopril..." [from learned distribution]
  Assessor: Detects error
  Injector reward: -1.0

Round 100:
  Injector: "Patient managed with enalapril..." [different wording, similar error]
  Assessor: Maybe misses it
  Injector reward: +1.0

Result: Injector learns to generate more deceptive errors!
```

---

## Example: Minimal Paraphrasing

### Original Note
```
A 42-year-old woman is brought to the physician by her husband because of
a 1-year history of abnormal behavior. During this time she has been
irritable, restless, and has had multiple episodes of hearing voices.
Over the past month, she has also had difficulty swallowing. She has a
2-year history of depression. She was let go by her employer 6 months ago
because she could no longer handle all her tasks and often forgot about
assignments. Her father committed suicide at the age of 50.
```

### CORRECT (Minimal Paraphrase - Change 1 Sentence)
```
A 42-year-old woman is brought to the physician by her husband because of
a 1-year history of abnormal behavior. Throughout this period she has been
irritable, restless, and has had multiple episodes of hearing voices.
                            ↑ Only this sentence changed
Over the past month, she has also had difficulty swallowing. She has a
2-year history of depression. She was let go by her employer 6 months ago
because she could no longer handle all her tasks and often forgot about
assignments. Her father committed suicide at the age of 50.
```

**Easy to verify:** Just check the changed sentence preserves facts!

### INCORRECT (MEDEC AS-IS)
```
[Exact copy of MEDEC incorrect_note]
...
Suspected of Creutzfeldt-Jakob disease.
            ↑ Doctor-verified error (should be Huntington)
...
```

+ **Reasoning:** "Family history of suicide at 50 suggests genetic
Huntington (autosomal dominant). CJD is rapid (weeks-months), but patient
has 1-year insidious progression. Key clue: choreiform movements +
psychiatric symptoms + family history."

---

## Production Generation

### Test Run (3 pairs)
```bash
python scripts/generate_sft_data_safe.py \
    --input-jsonl data_processed/medec_paired/train_val_split/sft_train.jsonl \
    --output-dir data/sft_minimal_test \
    --prompt-file configs/prompts/gpt4o_medec_safe_augmentation.json \
    --num-pairs 3 \
    --add-reasoning \
    --openai-api-key $OPENAI_API_KEY
```

**Expected:** 6 examples (3 CORRECT + 3 INCORRECT), ~$0.10

### Production Run (500 pairs)
```bash
python scripts/generate_sft_data_safe.py \
    --input-jsonl data_processed/medec_paired/train_val_split/sft_train.jsonl \
    --output-dir data/sft_production \
    --prompt-file configs/prompts/gpt4o_medec_safe_augmentation.json \
    --num-pairs 500 \
    --add-reasoning \
    --openai-api-key $OPENAI_API_KEY
```

**Expected:** 1000 examples (500 CORRECT + 500 INCORRECT), ~$20-25

---

## Quality Checks

### For CORRECT (Minimal Paraphrase)
- [ ] Only 1 sentence changed?
- [ ] All medications preserved exactly?
- [ ] All lab values preserved exactly?
- [ ] All diagnoses preserved exactly?
- [ ] Changed sentence still medically accurate?

### For INCORRECT (MEDEC AS-IS)
- [ ] Identical to MEDEC incorrect_note? (run diff)
- [ ] Error location correctly identified?
- [ ] Reasoning explains WHY error is wrong?
- [ ] Reasoning mentions key clinical clues?
- [ ] Clinical impact stated?

---

## Training Pipeline

### Phase 1: SFT Training (Teaches Format)
```bash
# Generate data
python scripts/generate_sft_data_safe.py \
    --input-jsonl data_processed/medec_paired/train_val_split/sft_train.jsonl \
    --output-dir data/sft_production \
    --prompt-file configs/prompts/gpt4o_medec_safe_augmentation.json \
    --num-pairs 500 \
    --add-reasoning \
    --openai-api-key $OPENAI_API_KEY

# Train Qwen3-4B
python src/training/train_serl.py \
    --phase sft \
    --model-name Qwen/Qwen3-4B \
    --train-data data/sft_production/sft_combined_safe.jsonl \
    --save-path outputs/qwen3_4b_sft \
    --num-epochs 3 \
    --batch-size 4 \
    --learning-rate 5e-5
```

### Phase 2: RL Self-Play Training (Drives Creativity)
```bash
# Now model generates creative errors via RL
bash scripts/train_medserl_reinforce_pp.sh
```

**RL handles deception:**
- Fresh data each rollout (no memorization)
- Policy gradient pushes toward fooling assessor
- Model explores error variations naturally
- Self-play stays competitive

---

## Cost Analysis

### Minimal Paraphrasing (1 sentence)
- **CORRECT generation:** ~1,500 tokens per example
- **Reasoning generation:** ~1,200 tokens per example
- **Total per pair:** ~2,700 tokens

### For 500 pairs (1000 examples)
- **Total tokens:** ~1.35M tokens
- **Cost:** ~$20-25
- **Time:** ~1-2 hours (with rate limiting)

### Comparison to Heavy Paraphrasing
- Heavy: ~2M tokens, ~$35-40
- Minimal: ~1.35M tokens, ~$20-25
- **Savings:** ~40% cheaper, easier to verify

---

## Summary

✅ **Minimal paraphrasing (1 sentence)** for CORRECT examples
✅ **MEDEC errors AS-IS** for INCORRECT examples
✅ **Optional reasoning** for educational value
✅ **50/50 split** for calibration
✅ **Easy verification** - only check changed sentence
✅ **RL handles creativity** - model explores during self-play
✅ **Cost-effective** - ~40% cheaper than heavy paraphrasing

**Key Takeaway:** SFT teaches format, RL drives deception. Keep SFT simple!
