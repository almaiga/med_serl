# MedSeRL Architecture Comparison: Offline vs Online RL

## Visual Comparison

### ❌ OLD: Offline RL (Two-Phase Approach)

```
┌───────────────────────────────────────────────────────────────────┐
│                    PHASE 1: DATA GENERATION                       │
│                      (Separate Script)                            │
└───────────────────────────────────────────────────────────────────┘

Input: MEDEC Training Set (2000 notes)
   │
   ▼
┌─────────────────────────────────────┐
│  quick_infer_qwen3_4b_lora.py      │
│  - Load model                       │
│  - For each note:                   │
│    1. Generate with injector prompt │
│    2. Apply VCF filters             │
│    3. If pass: save to JSONL        │
│    4. If fail: retry (max 3x)       │
└─────────────────────────────────────┘
   │
   ▼
Output: selfplay_data_vcf_filtered.jsonl (500-800 samples)
   │
   │ ⚠️ DISK I/O - Static Dataset Created
   ▼

┌───────────────────────────────────────────────────────────────────┐
│                    PHASE 2: RL TRAINING                           │
│                  (OpenRLHF Training)                              │
└───────────────────────────────────────────────────────────────────┘

Load: selfplay_data_vcf_filtered.jsonl
   │
   ▼
┌─────────────────────────────────────┐
│  REINFORCE++ Training Loop          │
│  - Load static filtered dataset     │
│  - For each epoch:                  │
│    - Shuffle data                   │
│    - For each batch:                │
│      1. Forward pass                │
│      2. Compute rewards             │
│      3. Policy update               │
└─────────────────────────────────────┘
   │
   ▼
Output: Model checkpoints

⚠️ PROBLEMS:
- Distribution shift (data from old policy)
- No fresh attacks (stale adversarial examples)
- Inefficient (disk I/O overhead)
- Two separate scripts to maintain
```

---

### ✅ NEW: Online RL (Integrated Approach)

```
┌───────────────────────────────────────────────────────────────────┐
│                ONLINE RL TRAINING ROUND (Repeat)                  │
│                  (Single Integrated Script)                       │
└───────────────────────────────────────────────────────────────────┘

Input: MEDEC Training Set (raw notes)
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: SAMPLE BATCH (32/64/128 notes)                         │
│  - Random sample from MEDEC training set                        │
│  - Fresh batch every round                                      │
└─────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: INJECTOR ROLLOUT (Frozen/Generation Phase)             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  vLLM Generation Engine                                   │  │
│  │  - Model: Current policy (frozen for rollout)             │  │
│  │  - Prompt: Injector role instructions                     │  │
│  │  - Generate modified notes (batch parallel)               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  VCF Inline Filtering                                     │  │
│  │  For each generated note:                                 │  │
│  │    1. Check Jaccard similarity (0.85-0.99)                │  │
│  │    2. Check word edit count (≤6)                          │  │
│  │    3. Check single error constraint                       │  │
│  │    4. If REJECT: Regenerate (max 3 attempts)              │  │
│  │    5. If ACCEPT: Continue to assessor                     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: ASSESSOR ROLLOUT (Evaluation Phase)                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  vLLM Generation Engine                                   │  │
│  │  - Model: Same policy (frozen for rollout)                │  │
│  │  - Prompt: Assessor role instructions                     │  │
│  │  - Input: VCF-accepted modified notes                     │  │
│  │  - Output: CORRECT / INCORRECT predictions                │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: COMPUTE ZERO-SUM REWARDS                               │
│  - Assessor correct detection: +1.0                             │
│  - Assessor fooled by injector: Injector wins                   │
│  - Structural bonus: +0.1 for <think> tags                      │
│  - False positive penalty: -1.5                                 │
│  - False negative penalty: -1.0                                 │
└─────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 5: POLICY UPDATE (REINFORCE++)                            │
│  - Gradient ascent: ∇θ J(θ) = Σ ∇θ log π(a|s) * R              │
│  - KL penalty vs reference model: -β * KL(π || π_ref)          │
│  - Update both injector and assessor paths                      │
└─────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 6: SAMPLE NEW BATCH → REPEAT                              │
│  - Sample different 32/64/128 notes                             │
│  - Continue training with updated policy                        │
└─────────────────────────────────────────────────────────────────┘

✅ BENEFITS:
- No distribution shift (always on-policy)
- Fresh adversarial attacks every round
- Efficient (no disk I/O between steps)
- Single integrated script
- Faster convergence
```

---

## Detailed Flow Comparison

### Data Flow: Offline RL

```
MEDEC Raw Data
    ↓
[Generate Script] quick_infer_qwen3_4b_lora.py
    ↓
Apply VCF Filters (one-time)
    ↓
Save to JSONL (disk write)
    ↓
Load JSONL (disk read)
    ↓
[Training Script] train_ppo_ray
    ↓
Iterate over static dataset
    ↓
Policy updates on stale data
    ↓
Checkpoints

⏱️ Timeline: Generation (hours) → Disk I/O → Training (hours)
📦 Storage: Large JSONL files (GB)
🔄 Freshness: Data becomes stale as policy improves
```

### Data Flow: Online RL

```
MEDEC Raw Data
    ↓
[Training Script] medserl_trainer.py
    ├─ Sample batch
    ├─ Injector rollout + VCF inline
    ├─ Assessor rollout
    ├─ Compute rewards
    ├─ Policy update
    └─ Repeat with new batch
    ↓
Checkpoints

⏱️ Timeline: Training (hours) - everything inline
📦 Storage: Only checkpoints (no intermediate data)
🔄 Freshness: Data generated fresh each round
```

---

## Code Structure Comparison

### OLD: Two Separate Scripts

**File 1: `scripts/sft/quick_infer_qwen3_4b_lora.py` (1,242 lines)**
```python
def collect_selfplay_data(...):
    """Pre-generate VCF-filtered dataset."""
    model = load_model(...)
    filtered_samples = []

    for note in tqdm(notes):
        for attempt in range(max_retries):
            # Generate
            output = model.generate(injector_prompt(note))

            # Apply VCF
            filter_result = apply_vcf(note, output, ...)

            if filter_result.passed:
                filtered_samples.append({
                    "input": note,
                    "output": output,
                    "filter_meta": filter_result,
                })
                break

    # Save to disk
    write_jsonl(filtered_samples, "selfplay_data_vcf_filtered.jsonl")
```

**File 2: `scripts/train_medserl_reinforce_pp.sh`**
```bash
# Load pre-filtered data
prompt_data=selfplay_data_vcf_filtered.jsonl

ray job submit ... \
    --prompt_data $prompt_data \
    ...
```

⚠️ **Problems**:
- Two scripts to maintain
- Manual coordination between them
- Data gets stale as policy improves
- Disk I/O overhead

---

### NEW: Single Integrated Script

**File: `src/training/medserl_trainer.py`**
```python
class MedSeRLTrainer(PPOTrainer):
    """Custom PPO trainer with inline VCF."""

    def __init__(self, *args, vcf_config: Dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.vcf_rollout = VCFRolloutGenerator(
            vllm_engine=self.vllm_engines[0],
            tokenizer=self.tokenizer,
            vcf_config=vcf_config,
        )

    def training_step(self, batch: Dict) -> Dict:
        """Single training step with inline VCF."""

        # Step 1: Injector rollout + VCF inline
        injector_prompts = self.build_injector_prompts(batch["notes"])
        injector_outputs, vcf_results = self.vcf_rollout.generate_with_vcf(
            prompts=injector_prompts,
            original_notes=batch["notes"],
            sampling_params=self.sampling_params,
        )

        # Step 2: Assessor rollout
        assessor_prompts = self.build_assessor_prompts(injector_outputs)
        assessor_outputs = self.vllm_engines[0].generate(
            assessor_prompts,
            sampling_params=self.sampling_params,
        )

        # Step 3: Compute rewards
        rewards = self.compute_rewards(
            injector_outputs,
            assessor_outputs,
            ground_truth=batch["labels"],
        )

        # Step 4: Policy update (REINFORCE++)
        loss = self.ppo_step(
            queries=batch["input_ids"],
            responses=injector_outputs + assessor_outputs,
            rewards=rewards,
        )

        return {
            "loss": loss,
            "mean_reward": rewards.mean(),
            "vcf_acceptance_rate": sum(r.passed for r in vcf_results) / len(vcf_results),
        }
```

**File: `src/training/vcf_rollout.py`**
```python
class VCFRolloutGenerator:
    """VCF-aware rollout generator for inline filtering."""

    def generate_with_vcf(
        self,
        prompts: List[str],
        original_notes: List[str],
        sampling_params: Dict,
    ) -> Tuple[List[str], List[FilterResult]]:
        """Generate with inline VCF filtering."""
        accepted_outputs = []
        filter_results = []

        for prompt, original_note in zip(prompts, original_notes):
            # Retry loop for VCF
            for attempt in range(self.max_retries):
                # Generate
                output = self.vllm_engine.generate(
                    prompt,
                    sampling_params=sampling_params
                )[0].outputs[0].text

                # Apply VCF inline
                generated_note = extract_generated_note(output)
                filter_result = apply_vcf(
                    original_note,
                    generated_note,
                    self.vcf_config,
                )

                if filter_result.passed:
                    accepted_outputs.append(output)
                    filter_results.append(filter_result)
                    break

                # Retry on rejection
                if attempt == self.max_retries - 1:
                    # Max retries - use last attempt anyway
                    accepted_outputs.append(output)
                    filter_results.append(filter_result)

        return accepted_outputs, filter_results
```

✅ **Benefits**:
- Single integrated script
- VCF happens inline (no separate step)
- Fresh data every batch
- No disk I/O overhead

---

## Performance Comparison

### OLD: Offline RL

| Metric | Value | Issue |
|--------|-------|-------|
| Data generation time | 2-4 hours | Blocking (must finish before training) |
| Disk I/O overhead | ~1-2 GB/dataset | Slow read/write |
| Distribution shift | High | Data from old policy |
| Training efficiency | Low | Stale adversarial examples |
| Convergence speed | Slow | Offline RL limitation |

### NEW: Online RL

| Metric | Value | Benefit |
|--------|-------|---------|
| Data generation time | 0 (inline) | No blocking |
| Disk I/O overhead | 0 (in-memory) | Fast |
| Distribution shift | None | On-policy data |
| Training efficiency | High | Fresh adversarial examples |
| Convergence speed | Fast | Online RL advantage |

---

## Hyperparameters

### Batch Sizes

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `rollout_batch_size` | 64 | Number of notes per training round |
| `train_batch_size` | 16 | Gradient accumulation batch size |
| `micro_train_batch_size` | 2 | Per-device batch size |

**Training Round Flow**:
```
Sample 64 notes
  ↓
Injector generates 64 modified notes (with VCF retry)
  ↓
Assessor evaluates 64 notes
  ↓
Compute 64 rewards
  ↓
Policy update with train_batch_size=16 (4 gradient steps)
```

### VCF Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `vcf_min_jaccard` | 0.85 | Minimum similarity threshold |
| `vcf_max_jaccard` | 0.99 | Maximum similarity threshold |
| `vcf_max_word_edits` | 6 | Maximum word changes allowed |
| `vcf_max_retries` | 3 | Retry attempts on VCF rejection |

---

## Summary

**Key Takeaway**: MedSeRL should use **online batch-based RL** where VCF filtering happens inline during training rollouts, NOT as a separate offline data generation step.

This ensures:
- ✅ True adversarial self-play with fresh attacks
- ✅ No distribution shift from stale data
- ✅ Faster iteration and better convergence
- ✅ More efficient training pipeline (no disk I/O)
- ✅ Single integrated script (easier to maintain)

**The revised architecture is now ready for implementation.**
