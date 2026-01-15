# MedSeRL Testing Guide

Complete guide for testing the MedSeRL implementation at different levels.

---

## Quick Test Summary

| Test Level | Time | GPU Required | What It Tests |
|------------|------|--------------|---------------|
| 1. Core Logic | 1 min | ❌ No | VCF filtering logic only |
| 2. Full VCF (with torch) | 2 min | ❌ No | VCF + inference_utils integration |
| 3. Mock Training Loop | 5 min | ❌ No | Trainer structure without real model |
| 4. Single Batch (real model) | 10 min | ✅ Yes | One training round with small model |
| 5. Mini Training Run | 30 min | ✅ Yes | 10 rounds with small dataset |
| 6. Full Training Run | 2-4 hours | ✅ Yes | Complete training pipeline |

---

## Test 1: Core VCF Logic (No Dependencies)

**What**: Tests Jaccard similarity and basic filtering logic
**Time**: 1 minute
**Requirements**: Just Python 3

```bash
python3 scripts/test_vcf_simple.py
```

**Expected Output**:
```
============================================================
VCF Core Logic Tests (No torch required)
============================================================

Test 1: Single word change
  Jaccard: 0.778 (expected: ~0.91)
  Has change: True
  Would pass VCF: False

✅ Core VCF logic tests complete!
```

---

## Test 2: Full VCF with Inference Utils (With torch)

**What**: Tests complete VCF pipeline including word-level diff analysis
**Time**: 2 minutes
**Requirements**: torch, transformers installed

```bash
# In your training environment with torch
python3 scripts/test_vcf.py
```

**Expected Output**:
```
============================================================
Test 1: Basic VCF Filtering
============================================================

Original: Patient has type 2 diabetes controlled with metformin.
Modified: Patient has type 1 diabetes controlled with metformin.
Result: True (expected: True)
  Jaccard: 0.778
  Word edits: 2
✅ Test 1 passed

...

✅ All tests passed!
VCF filtering is working correctly.
```

---

## Test 3: Mock Training Loop (No GPU Needed)

**What**: Tests trainer structure and logging without real model inference
**Time**: 5 minutes
**Requirements**: Python 3, basic libraries

Create a mock test:

```bash
cat > scripts/test_trainer_mock.py << 'EOF'
#!/usr/bin/env python3
"""
Mock test for MedSeRL trainer structure without real model.
Tests logging, batch processing, reward calculation.
"""

import json
import os
import sys
import tempfile
from datetime import datetime

# Mock data
class MockBatch:
    def __init__(self):
        self.data = {
            "notes": [
                "Patient has type 2 diabetes.",
                "Patient presents with chest pain.",
            ],
            "labels": ["INCORRECT", "INCORRECT"],
        }

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

# Mock VCF result
class MockFilterResult:
    def __init__(self, passed=True):
        self.passed = passed
        self.score_jaccard = 0.87
        self.reason = None if passed else "too_many_edits"
        self.word_edits = 2
        self.sentences_changed = 1

# Test logging
print("=" * 60)
print("Mock Trainer Test")
print("=" * 60)

# Create temp log files
with tempfile.TemporaryDirectory() as tmpdir:
    interaction_log_path = os.path.join(tmpdir, "interactions.jsonl")
    metrics_log_path = os.path.join(tmpdir, "metrics.jsonl")

    # Simulate training round
    print("\n[Test] Simulating training round...")

    batch = MockBatch()
    round_num = 0

    # Mock injector outputs
    injector_outputs = [
        "<think>Change type 2 to type 1</think>\nPatient has type 1 diabetes.",
        "<think>Change chest to back</think>\nPatient presents with back pain.",
    ]

    # Mock assessor outputs
    assessor_outputs = [
        "<think>Type 1 with no insulin mentioned - error!</think>\nANSWER: INCORRECT",
        "<think>Back pain is different from chest pain</think>\nANSWER: CORRECT",
    ]

    # Mock VCF results
    vcf_results = [MockFilterResult(passed=True), MockFilterResult(passed=True)]

    # Mock rewards
    rewards = [1.0, -1.0]  # First: assessor wins, Second: injector wins

    # Write interaction log
    print("[Test] Writing interaction log...")
    with open(interaction_log_path, 'a') as f:
        for i in range(len(batch["notes"])):
            interaction = {
                "round": round_num,
                "sample_idx": i,
                "original_note": batch["notes"][i],
                "injector_output": injector_outputs[i],
                "assessor_output": assessor_outputs[i],
                "reward": rewards[i],
                "vcf_passed": vcf_results[i].passed,
                "vcf_jaccard": vcf_results[i].score_jaccard,
            }
            f.write(json.dumps(interaction) + "\n")

    # Write metrics log
    print("[Test] Writing metrics log...")
    with open(metrics_log_path, 'a') as f:
        metrics = {
            "round": round_num,
            "loss": 0.234,
            "mean_reward": sum(rewards) / len(rewards),
            "vcf_acceptance_rate": sum(r.passed for r in vcf_results) / len(vcf_results),
            "injector_win_rate": sum(1 for r in rewards if r < 0) / len(rewards),
            "assessor_win_rate": sum(1 for r in rewards if r > 0) / len(rewards),
            "timestamp": datetime.utcnow().isoformat(),
        }
        f.write(json.dumps(metrics) + "\n")

    # Verify logs
    print("[Test] Verifying logs...")

    with open(interaction_log_path) as f:
        interactions = [json.loads(line) for line in f]

    with open(metrics_log_path) as f:
        metrics = [json.loads(line) for line in f]

    print(f"✅ Interactions logged: {len(interactions)}")
    print(f"✅ Metrics logged: {len(metrics)}")

    # Show sample interaction
    print("\nSample Interaction:")
    print(json.dumps(interactions[0], indent=2))

    # Show metrics
    print("\nMetrics:")
    print(json.dumps(metrics[0], indent=2))

print("\n" + "=" * 60)
print("✅ Mock trainer test passed!")
print("=" * 60)
EOF

python3 scripts/test_trainer_mock.py
```

**Expected Output**:
```
============================================================
Mock Trainer Test
============================================================

[Test] Simulating training round...
[Test] Writing interaction log...
[Test] Writing metrics log...
[Test] Verifying logs...
✅ Interactions logged: 2
✅ Metrics logged: 1

✅ Mock trainer test passed!
```

---

## Test 4: Single Batch with Real Model (GPU Required)

**What**: Tests one complete training round with a real small model
**Time**: 10 minutes
**Requirements**: GPU, torch, transformers, vLLM (optional)

```bash
cat > scripts/test_single_batch.py << 'EOF'
#!/usr/bin/env python3
"""
Test single training batch with real model (small/fast model recommended).
"""

import sys
import os
sys.path.insert(0, 'scripts/sft')
sys.path.insert(0, 'src/training')

from inference_utils import apply_vcf
import torch

print("=" * 60)
print("Single Batch Test with Real Model")
print("=" * 60)

# Test data
original_notes = [
    "Patient has type 2 diabetes controlled with metformin.",
    "Patient presents with acute chest pain radiating to left arm.",
]

# Simulate injector generation (in real training, this uses vLLM)
print("\n[Step 1] Simulating injector generation...")
generated_notes = [
    "Patient has type 1 diabetes controlled with metformin.",  # Changed type 2 -> type 1
    "Patient presents with acute chest pain radiating to right leg.",  # Changed left arm -> right leg
]

# Test VCF filtering
print("[Step 2] Applying VCF filters...")
for i, (orig, gen) in enumerate(zip(original_notes, generated_notes)):
    result = apply_vcf(orig, gen, min_jaccard=0.85, max_jaccard=0.99, max_word_edits=6)
    print(f"\n  Sample {i+1}:")
    print(f"    Original: {orig[:50]}...")
    print(f"    Generated: {gen[:50]}...")
    print(f"    VCF Result: {'✅ PASS' if result.passed else '❌ REJECT'}")
    print(f"    Jaccard: {result.score_jaccard:.3f}")
    print(f"    Reason: {result.reason if result.reason else 'N/A'}")

print("\n[Step 3] Simulating assessor evaluation...")
assessor_predictions = ["INCORRECT", "CORRECT"]  # Assessor response
ground_truth = ["INCORRECT", "INCORRECT"]

# Compute rewards
print("[Step 4] Computing rewards...")
for i in range(len(original_notes)):
    correct = assessor_predictions[i] == ground_truth[i]
    reward = 1.0 if correct else -1.0
    print(f"  Sample {i+1}: Reward = {reward:.1f} ({'Assessor wins' if reward > 0 else 'Injector wins'})")

print("\n" + "=" * 60)
print("✅ Single batch test complete!")
print("=" * 60)
print("\nNext: Try test_mini_training.py for a full training loop")
EOF

python3 scripts/test_single_batch.py
```

---

## Test 5: Mini Training Run (10 Rounds)

**What**: Small-scale training with 10 rounds to verify full pipeline
**Time**: 30 minutes
**Requirements**: GPU, full training environment

```bash
# Create mini dataset (10 notes)
head -10 data/medec_train.jsonl > data/medec_train_mini.jsonl

# Run mini training
export ROLLOUT_BATCH_SIZE=4         # Small batch
export MAX_EPOCHS=1                 # Just 1 epoch
export SAVE_STEPS=5                 # Save every 5 steps

bash scripts/train_medserl_reinforce_pp.sh \
    --prompt-data data/medec_train_mini.jsonl \
    --save-path outputs/test_mini \
    --logging-dir outputs/test_mini/logs

# Monitor progress
tail -f outputs/test_mini/logs/metrics.jsonl | jq '{round, loss, injector_win_rate}'
```

**Expected Output**:
```
[INFO] Using custom MedSeRL trainer
[Step 1/6] Loading model...
[Step 2/6] Initializing VCF rollout...
[Step 3/6] Starting training...

Round 0: loss=0.234, mean_reward=0.12, vcf_acceptance=0.87
Round 1: loss=0.221, mean_reward=0.15, vcf_acceptance=0.85
...
Round 9: loss=0.189, mean_reward=0.23, vcf_acceptance=0.83

✅ Mini training complete!
```

---

## Test 6: Full Training Run (Production)

**What**: Complete training with full dataset
**Time**: 2-4 hours
**Requirements**: GPU(s), full training environment

```bash
# Start Ray cluster
ray start --head --port=6379

# Launch full training
bash scripts/train_medserl_reinforce_pp.sh

# Monitor in real-time (3 terminals)

# Terminal 1: Interactions
tail -f outputs/medserl_*/logs/interactions.jsonl | jq '{round, injector_won, assessor_won, vcf_passed}'

# Terminal 2: Metrics
tail -f outputs/medserl_*/logs/metrics.jsonl | jq '{round, loss, injector_win_rate, vcf_acceptance_rate}'

# Terminal 3: Convergence check
watch -n 10 "tail -50 outputs/medserl_*/logs/metrics.jsonl | jq -s 'map(.injector_win_rate) | add / length'"
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Solution**: Run test in environment with torch installed, or use Test 1 (core logic only)

```bash
# Option 1: Install torch
pip install torch transformers

# Option 2: Use simple test
python3 scripts/test_vcf_simple.py
```

### Issue: "Cannot import inference_utils"

**Solution**: Make sure you're running from the repository root

```bash
cd /Users/josmaiga/.claude-worktrees/sft/pedantic-hertz
python3 scripts/test_vcf.py
```

### Issue: VCF acceptance rate too low (<50%)

**Solution**: Relax VCF thresholds

```bash
export VCF_MIN_JACCARD=0.80  # Lower from 0.85
export VCF_MAX_JACCARD=0.95  # Lower from 0.99
```

### Issue: Ray cluster not starting

**Solution**: Check Ray installation and ports

```bash
# Install Ray
pip install ray

# Check if port 6379 is available
lsof -i :6379

# Start with different port
ray start --head --port=6380
```

---

## Recommended Testing Order

For first-time testing, follow this order:

1. ✅ **Test 1: Core Logic** (1 min, no dependencies)
   - Verify VCF filtering logic works

2. ✅ **Test 2: Full VCF** (2 min, needs torch)
   - Verify complete VCF pipeline with word-level diff

3. ✅ **Test 3: Mock Trainer** (5 min, no GPU)
   - Verify trainer structure and logging

4. ⏳ **Test 4: Single Batch** (10 min, needs GPU)
   - Verify one training round with real model

5. ⏳ **Test 5: Mini Training** (30 min, needs GPU)
   - Verify full pipeline with small dataset

6. ⏳ **Test 6: Full Training** (2-4 hours, needs GPU)
   - Production training run

---

## Success Criteria

### Test 1-3 (Structure Tests)
- ✅ No errors or exceptions
- ✅ VCF correctly accepts/rejects test cases
- ✅ Log files created with correct format

### Test 4-5 (Integration Tests)
- ✅ VCF acceptance rate: 70-90%
- ✅ Rewards computed correctly (mix of positive/negative)
- ✅ Logs written in real-time (can tail -f)
- ✅ No memory leaks or crashes

### Test 6 (Production Run)
- ✅ Training completes without errors
- ✅ VCF acceptance rate stable (not dropping)
- ✅ Injector win rate converges toward 0.5
- ✅ Checkpoints saved correctly
- ✅ Metrics show learning progress

---

## Next Steps After Testing

Once tests pass:

1. **Run evaluation on MEDEC test sets**
2. **Analyze convergence curves**
3. **Compare with baseline (no VCF)**
4. **Tune hyperparameters** (learning rate, VCF thresholds)
5. **Scale to full dataset** (if using mini for testing)

---

## Quick Commands Reference

```bash
# Test 1: Core logic only
python3 scripts/test_vcf_simple.py

# Test 2: Full VCF (needs torch)
python3 scripts/test_vcf.py

# Test 3: Mock trainer
python3 scripts/test_trainer_mock.py

# Test 4: Single batch (needs GPU + model)
python3 scripts/test_single_batch.py

# Test 5: Mini training (10 rounds)
bash scripts/train_medserl_reinforce_pp.sh --prompt-data data/medec_train_mini.jsonl

# Test 6: Full training
bash scripts/train_medserl_reinforce_pp.sh

# Monitor training
tail -f outputs/medserl_*/logs/metrics.jsonl | jq '{round, loss, injector_win_rate}'
```
