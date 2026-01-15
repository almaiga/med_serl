# MedSeRL Implementation Plan - Revision Summary

## Date: 2026-01-14

## Executive Summary

The MedSeRL implementation plan has been **revised to reflect batch-based online RL training** instead of offline data pre-filtering. This critical correction ensures the implementation matches the actual MedSeRL algorithm design.

---

## What Changed?

### ❌ Original Plan (Incorrect - Offline RL)

```
Step 1: Pre-generate VCF-filtered data
  ↓
  Use quick_infer_qwen3_4b_lora.py to generate 1000s of samples
  Apply VCF filters, save to selfplay_data_vcf_filtered.jsonl

Step 2: Train REINFORCE++ on pre-collected data
  ↓
  Load filtered JSONL file
  Train policy on static dataset
  Update weights after full dataset
```

### ✅ Revised Plan (Correct - Online RL)

```
Training Round (Repeat):
  ↓
Step 1: Sample batch of notes (32/64/128) from MEDEC training set
  ↓
Step 2: INJECTOR rollout (frozen/generation phase)
  → Generate modified notes
  → Apply VCF filtering INLINE (max 3 retry attempts)
  → Only VCF-accepted samples proceed
  ↓
Step 3: ASSESSOR rollout (evaluation phase)
  → Same model evaluates generated notes
  → Predicts CORRECT/INCORRECT
  ↓
Step 4: Compute zero-sum rewards
  → Assessor correct detection: +1.0
  → Assessor fooled: Injector wins
  ↓
Step 5: POLICY UPDATE (REINFORCE++)
  → Gradient ascent on policy
  → KL penalty vs reference model
  ↓
Step 6: Sample NEW batch → Repeat
```

---

## Key Differences

| Aspect | Old (Offline RL) | New (Online RL) |
|--------|------------------|-----------------|
| **Training Type** | Offline - pre-filtered static data | Online - fresh batches each round |
| **VCF Timing** | Pre-processing step (separate script) | Inline during rollouts (integrated) |
| **Data Reuse** | Same filtered dataset repeatedly | New batch every round |
| **Policy Updates** | After full dataset | After each batch (32/64/128) |
| **Retry Logic** | N/A (data already filtered) | Max 3 VCF retry attempts per sample |
| **Injector Phase** | N/A (data pre-generated) | Frozen rollout with current policy |
| **Code Structure** | Heavy reuse of quick_infer | Custom OpenRLHF trainer extension |

---

## Why This Matters

### 1. **True Adversarial Self-Play**
- **Old**: Model trained on stale, pre-generated attacks
- **New**: Model generates fresh attacks each round, forcing continuous adaptation

### 2. **No Distribution Shift**
- **Old**: VCF filters on initial policy's outputs (distribution mismatch over time)
- **New**: VCF filters current policy's outputs (always on-distribution)

### 3. **Faster Iteration**
- **Old**: Must wait for full dataset generation before any training
- **New**: Policy updates immediately after each batch

### 4. **More Efficient**
- **Old**: Store/load large filtered datasets (disk I/O overhead)
- **New**: No intermediate storage, direct batch processing

### 5. **Better Convergence**
- **Old**: Offline RL typically slower, can suffer from extrapolation errors
- **New**: Online RL converges faster, more stable training dynamics

---

## Implementation Changes

### Phase 1: VCF Integration

**Old Approach**:
```python
def collect_selfplay_data(...):
    """Pre-generate VCF-filtered dataset."""
    # Generate 1000s of samples using quick_infer
    # Apply VCF filters
    # Save to selfplay_data_vcf_filtered.jsonl
```

**New Approach**:
```python
class VCFRolloutGenerator:
    """Inline VCF filtering during rollouts."""
    def generate_with_vcf(self, prompts, original_notes, ...):
        # Generate with vLLM
        # Apply VCF immediately
        # Retry on rejection (max 3 attempts)
        # Return only accepted samples
```

### Phase 2: Training Script

**Old Approach**:
```bash
# Step 1: Generate filtered data
python quick_infer_qwen3_4b_lora.py --selfplay ...

# Step 2: Train on pre-filtered data
python train_ppo_ray.py --prompt-data selfplay_vcf_filtered.jsonl
```

**New Approach**:
```bash
# Direct training with inline VCF (no pre-generation)
python medserl_trainer.py \
    --prompt-data medec_train.jsonl \
    --vcf-min-jaccard 0.85 \
    --vcf-max-jaccard 0.99 \
    --vcf-max-word-edits 6 \
    --rollout-batch-size 64
```

### Phase 5: End-to-End Pipeline

**Old Approach**:
```bash
# Step 1: Generate VCF-filtered self-play data
python quick_infer_qwen3_4b_lora.py --selfplay ...

# Step 2: REINFORCE++ training
bash train_medserl_reinforce_pp.sh --prompt-data selfplay_vcf_filtered.jsonl
```

**New Approach**:
```bash
# Direct online RL training (VCF happens inline)
bash train_medserl_reinforce_pp.sh \
    --prompt-data medec_train.jsonl \
    --vcf-min-jaccard 0.85
# No pre-generation step needed
```

---

## Critical Files (Updated)

| File | Purpose | Status | Change |
|------|---------|--------|--------|
| `scripts/sft/inference_utils.py` | VCF utility functions | Modify | Add enhanced VCF with single error check |
| `src/training/vcf_rollout.py` | **NEW** VCF-aware rollout generator | Create | Wraps vLLM with inline filtering |
| `src/training/medserl_trainer.py` | **NEW** Custom PPO trainer with VCF | Create | Extends OpenRLHF PPOTrainer |
| `scripts/train_medserl_reinforce_pp.sh` | Training launch script | Create | Batch-based online RL |
| `src/training/medec_reward.py` | Zero-sum reward function | Modify | Compute rewards based on role |
| `scripts/evaluate_medserl.py` | Evaluation pipeline | Create | MEDEC tests + convergence |
| `scripts/run_medserl_pipeline.sh` | End-to-end workflow | Create | Online RL pipeline |

**Removed**:
- ❌ No longer using `quick_infer_qwen3_4b_lora.py` for data generation
- ❌ No `collect_selfplay_data()` function
- ❌ No pre-filtered JSONL datasets

---

## Verification Plan (Updated)

### 1. VCF Inline Filtering Test

```bash
# Unit test VCF functions
python -c "
from scripts.sft.inference_utils import apply_vcf

# Test single word change (should pass)
original = 'Patient has type 2 diabetes controlled with metformin.'
modified = 'Patient has type 1 diabetes controlled with metformin.'
result = apply_vcf(original, modified, min_jaccard=0.85, max_jaccard=0.99, max_word_edits=6)
assert result.passed, 'Single word change should pass VCF'
print('✓ VCF test passed')
"
```

### 2. Online Training Test

```bash
# Start Ray cluster
ray start --head --port=6379

# Launch online training (VCF happens inline)
bash scripts/train_medserl_reinforce_pp.sh \
    --prompt-data data/medec_train_sample.jsonl \
    --rollout-batch-size 32 \
    --max-epochs 1

# Monitor VCF acceptance rate in logs
tail -f outputs/medserl_*/logs/train.log | grep "vcf_acceptance_rate"
```

### 3. Batch Processing Verification

```python
# Verify batch training flow
from src.training.medserl_trainer import MedSeRLTrainer

trainer = MedSeRLTrainer(...)

# Simulate one training round
batch = sample_notes(size=64)  # Sample fresh batch
metrics = trainer.training_step(batch)

# Check metrics
assert 'vcf_acceptance_rate' in metrics
assert metrics['vcf_acceptance_rate'] > 0.1  # At least 10% pass
assert 'mean_reward' in metrics
print('✓ Batch training verified')
```

---

## User Feedback That Triggered Revision

> **User**: "There is a problem. The model receive a data batch 32/64/128 notes then first generate note first with injector froze then it play the assesor and policy update and then we start again witha new data batch. At the end of the round we sample different data and we do that again"

This clarified that MedSeRL should use **batch-based online RL** where:
- ✅ Batches sampled fresh each round (not pre-collected)
- ✅ Injector generates in frozen/rollout phase
- ✅ VCF filtering happens inline during generation
- ✅ Assessor evaluates immediately after
- ✅ Policy updates after each batch
- ✅ New batch sampled for next round

---

## Timeline (Unchanged)

- **Phase 1 (VCF Integration)**: 2-3 days
- **Phase 2 (REINFORCE++ Setup)**: 2-3 days
- **Phase 3 (Reward Integration)**: 1-2 days
- **Phase 4 (Evaluation)**: 1-2 days
- **Phase 5 (End-to-End Testing)**: 1 day

**Total**: ~7-11 days for complete implementation and validation

---

## Next Steps

1. **Implement Phase 1**: Create `VCFRolloutGenerator` class in `src/training/vcf_rollout.py`
2. **Implement Phase 2**: Create `MedSeRLTrainer` class in `src/training/medserl_trainer.py`
3. **Test VCF inline**: Verify VCF filtering works during rollouts (not pre-processing)
4. **Test batch training**: Verify one training round works end-to-end
5. **Full training run**: Train for 5 epochs and evaluate on MEDEC test sets

---

## Key Takeaway

**This revision transforms MedSeRL from offline RL (train on pre-filtered data) to online RL (VCF filtering happens inline during training rollouts).**

This matches the actual MedSeRL algorithm design and ensures:
- True adversarial self-play with fresh attacks
- No distribution shift from stale data
- Faster iteration and better convergence
- More efficient training pipeline

**The revised plan is now ready for implementation.**
