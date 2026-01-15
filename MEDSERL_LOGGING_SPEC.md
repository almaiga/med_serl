# MedSeRL Training Logging Specification

## Overview

MedSeRL training produces **two streaming log files** that allow real-time monitoring of training progress:

1. **`interactions.jsonl`** - Detailed play-by-play of every injector-assessor interaction
2. **`metrics.jsonl`** - Aggregated round-level statistics for tracking convergence

Both files are written incrementally during training (flushed after each write), so you can monitor training progress with `tail -f`.

---

## File 1: Interaction Log (`interactions.jsonl`)

**Purpose**: Track individual adversarial interactions to understand self-play dynamics.

**Location**: `<output_dir>/logs/interactions.jsonl`

**Update Frequency**: One line per sample (64 lines per training round for batch_size=64)

### Format

Each line is a JSON object representing one injector-assessor interaction:

```jsonl
{"round": 0, "sample_idx": 0, "original_note": "Patient has type 2 diabetes controlled with metformin.", "injector_output": "<think>I'll change type 2 to type 1 to create a subtle error</think>\nPatient has type 1 diabetes controlled with metformin.", "injector_note": "Patient has type 1 diabetes controlled with metformin.", "assessor_output": "<think>Type 1 diabetes with metformin is incorrect - metformin is not used for type 1</think>\nANSWER: INCORRECT", "assessor_prediction": "INCORRECT", "ground_truth": "INCORRECT", "reward": 1.0, "vcf_passed": true, "vcf_reason": null, "vcf_jaccard": 0.93, "injector_won": false, "assessor_won": true}
{"round": 0, "sample_idx": 1, "original_note": "Patient presents with acute chest pain radiating to left arm.", "injector_output": "<think>I'll change 'left arm' to 'right leg' - atypical presentation</think>\nPatient presents with acute chest pain radiating to right leg.", "assessor_output": "<think>Chest pain radiating to right leg is unusual but possible</think>\nANSWER: CORRECT", "assessor_prediction": "CORRECT", "ground_truth": "INCORRECT", "reward": -1.0, "vcf_passed": true, "vcf_reason": null, "vcf_jaccard": 0.91, "injector_won": true, "assessor_won": false}
...
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `round` | int | Training round number (0-indexed) |
| `sample_idx` | int | Index within batch (0-63 for batch_size=64) |
| `original_note` | str | Original clinical note (ground truth correct) |
| `injector_output` | str | Full injector output (may include `<think>` tags) |
| `injector_note` | str | Extracted modified note (without think tags) |
| `assessor_output` | str | Full assessor output (may include `<think>` tags) |
| `assessor_prediction` | str | Extracted answer: "CORRECT" or "INCORRECT" |
| `ground_truth` | str | True label: "CORRECT" or "INCORRECT" |
| `reward` | float | Reward received (+1.0 for win, -1.0 for loss, +0.1 structural bonus) |
| `vcf_passed` | bool | Did injector output pass VCF filters? |
| `vcf_reason` | str\|null | Why VCF rejected (null if passed) |
| `vcf_jaccard` | float\|null | Jaccard similarity score (0.0-1.0) |
| `injector_won` | bool | True if injector fooled assessor (reward > 0) |
| `assessor_won` | bool | True if assessor detected error (reward < 0) |

### Use Cases

**1. Monitor Self-Play Dynamics**:
```bash
# Count injector wins vs assessor wins in last 1000 interactions
tail -1000 interactions.jsonl | jq -s '[.[] | select(.injector_won)] | length'
tail -1000 interactions.jsonl | jq -s '[.[] | select(.assessor_won)] | length'
```

**2. Track VCF Acceptance Rate**:
```bash
# Check VCF acceptance rate in recent interactions
tail -1000 interactions.jsonl | jq -s '[.[] | select(.vcf_passed)] | length / 1000'
```

**3. Find Interesting Interactions**:
```bash
# Find cases where injector fooled assessor
jq 'select(.injector_won == true)' interactions.jsonl | head -5

# Find VCF rejections
jq 'select(.vcf_passed == false)' interactions.jsonl | head -10
```

**4. Analyze Error Types**:
```bash
# Group by VCF rejection reason
jq -s 'group_by(.vcf_reason) | map({reason: .[0].vcf_reason, count: length})' interactions.jsonl
```

---

## File 2: Metrics Log (`metrics.jsonl`)

**Purpose**: Track aggregate training metrics per round for convergence analysis.

**Location**: `<output_dir>/logs/metrics.jsonl`

**Update Frequency**: One line per training round (after policy update)

### Format

Each line is a JSON object representing aggregated metrics for one round:

```jsonl
{"round": 0, "loss": 0.234, "mean_reward": 0.12, "vcf_acceptance_rate": 0.87, "injector_win_rate": 0.42, "assessor_win_rate": 0.58, "timestamp": "2026-01-14T12:34:56.789Z"}
{"round": 1, "loss": 0.221, "mean_reward": 0.15, "vcf_acceptance_rate": 0.85, "injector_win_rate": 0.45, "assessor_win_rate": 0.55, "timestamp": "2026-01-14T12:35:12.345Z"}
{"round": 2, "loss": 0.209, "mean_reward": 0.18, "vcf_acceptance_rate": 0.83, "injector_win_rate": 0.48, "assessor_win_rate": 0.52, "timestamp": "2026-01-14T12:35:28.901Z"}
...
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `round` | int | Training round number (0-indexed) |
| `loss` | float | Policy gradient loss (REINFORCE++) |
| `mean_reward` | float | Average reward across batch |
| `vcf_acceptance_rate` | float | Fraction of injector outputs that passed VCF (0.0-1.0) |
| `injector_win_rate` | float | Fraction where injector fooled assessor (0.0-1.0) |
| `assessor_win_rate` | float | Fraction where assessor detected error (0.0-1.0) |
| `timestamp` | str | ISO 8601 timestamp |

### Use Cases

**1. Monitor Training Progress**:
```bash
# Watch metrics in real-time
tail -f metrics.jsonl | jq '{round, loss, mean_reward, injector_win_rate}'
```

**2. Check Convergence**:
```bash
# Plot injector win rate over time (should converge toward 0.5)
jq -s 'map({round, injector_win_rate})' metrics.jsonl > convergence.json
```

**3. Detect Training Issues**:
```bash
# Check if VCF acceptance rate is dropping (model learning to bypass filters?)
jq -s 'map({round, vcf_acceptance_rate})' metrics.jsonl | tail -20

# Check if rewards are collapsing (all zeros)
jq 'select(.mean_reward == 0)' metrics.jsonl
```

**4. Generate Training Report**:
```bash
# Summary statistics for entire training run
jq -s '{
  total_rounds: length,
  final_loss: .[-1].loss,
  final_injector_win_rate: .[-1].injector_win_rate,
  final_vcf_acceptance: .[-1].vcf_acceptance_rate,
  avg_mean_reward: (map(.mean_reward) | add / length)
}' metrics.jsonl
```

---

## Real-Time Monitoring During Training

### Terminal 1: Launch Training
```bash
bash scripts/train_medserl_reinforce_pp.sh
```

### Terminal 2: Monitor Interactions
```bash
# Watch recent interactions
tail -f outputs/medserl_*/logs/interactions.jsonl | jq '{
  round,
  injector_won,
  assessor_won,
  vcf_passed,
  reward
}'
```

### Terminal 3: Monitor Metrics
```bash
# Watch round-level metrics
tail -f outputs/medserl_*/logs/metrics.jsonl | jq '{
  round,
  loss,
  injector_win_rate,
  vcf_acceptance_rate
}'
```

### Terminal 4: Convergence Check
```bash
# Check if converging to Nash equilibrium (win rate → 0.5)
watch -n 5 "tail -20 outputs/medserl_*/logs/metrics.jsonl | jq -s 'map(.injector_win_rate) | add / length'"
```

---

## Visualization Scripts

### Plot Convergence Curves

```python
import json
import matplotlib.pyplot as plt

# Load metrics
with open('outputs/medserl_*/logs/metrics.jsonl') as f:
    metrics = [json.loads(line) for line in f]

rounds = [m['round'] for m in metrics]
injector_win_rate = [m['injector_win_rate'] for m in metrics]
vcf_acceptance = [m['vcf_acceptance_rate'] for m in metrics]
loss = [m['loss'] for m in metrics]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Self-play convergence
axes[0, 0].plot(rounds, injector_win_rate, label='Injector Win Rate')
axes[0, 0].axhline(0.5, color='r', linestyle='--', label='Nash Equilibrium')
axes[0, 0].set_title('Self-Play Convergence')
axes[0, 0].set_xlabel('Round')
axes[0, 0].set_ylabel('Injector Win Rate')
axes[0, 0].legend()

# Plot 2: VCF acceptance rate
axes[0, 1].plot(rounds, vcf_acceptance)
axes[0, 1].set_title('VCF Acceptance Rate')
axes[0, 1].set_xlabel('Round')
axes[0, 1].set_ylabel('Acceptance Rate')

# Plot 3: Training loss
axes[1, 0].plot(rounds, loss)
axes[1, 0].set_title('Training Loss')
axes[1, 0].set_xlabel('Round')
axes[1, 0].set_ylabel('Loss')

# Plot 4: Mean reward
mean_reward = [m['mean_reward'] for m in metrics]
axes[1, 1].plot(rounds, mean_reward)
axes[1, 1].set_title('Mean Reward per Round')
axes[1, 1].set_xlabel('Round')
axes[1, 1].set_ylabel('Mean Reward')

plt.tight_layout()
plt.savefig('training_convergence.png')
print('Saved to training_convergence.png')
```

---

## Expected Patterns

### Healthy Training

**Early Rounds (0-100)**:
- `injector_win_rate`: ~0.3-0.4 (assessor initially stronger)
- `vcf_acceptance_rate`: ~0.8-0.9 (most generations pass)
- `mean_reward`: Slightly negative (assessor winning more)

**Mid Training (100-500)**:
- `injector_win_rate`: Rising toward 0.45-0.5
- `vcf_acceptance_rate`: Stable ~0.85
- `mean_reward`: Approaching 0

**Late Training (500+)**:
- `injector_win_rate`: Converging to 0.48-0.52 (Nash equilibrium)
- `vcf_acceptance_rate`: Stable ~0.80-0.90
- `mean_reward`: Oscillating around 0

### Warning Signs

**VCF Bypass**:
- `vcf_acceptance_rate` dropping below 0.5
- Indicates model learning to generate invalid outputs
- **Fix**: Increase VCF strictness or add more filters

**Reward Collapse**:
- `mean_reward` consistently 0
- Both roles performing poorly
- **Fix**: Check reward function, increase learning rate

**Assessor Dominance**:
- `assessor_win_rate` stuck above 0.7
- Injector not learning
- **Fix**: Increase injector thinking budget, check prompts

**Injector Dominance**:
- `injector_win_rate` stuck above 0.7
- Assessor not learning
- **Fix**: Increase assessor thinking budget, check prompts

---

## Log File Sizes

**Interactions Log**:
- ~1 KB per interaction
- 64 interactions per round (batch_size=64)
- ~64 KB per round
- ~32 MB for 500 rounds

**Metrics Log**:
- ~200 bytes per round
- ~100 KB for 500 rounds

**Total**: ~35 MB for 500 training rounds (manageable size)

---

## Summary

**Key Differences from Pre-Generation Approach**:

| Aspect | Old (Pre-Generate JSONL) | New (Streaming Logs) |
|--------|--------------------------|---------------------|
| **Purpose** | Store filtered training data | Monitor training in real-time |
| **Size** | 1-2 GB (full dataset) | 35 MB (500 rounds) |
| **When Created** | Before training | During training (streaming) |
| **Use Case** | Offline RL training input | Online RL monitoring |
| **Content** | Pre-filtered data samples | Live interaction logs + metrics |

**This logging approach enables**:
- ✅ Real-time monitoring of training progress
- ✅ Debugging self-play dynamics as they happen
- ✅ Early detection of training issues (VCF bypass, reward collapse)
- ✅ Post-training analysis and visualization
- ✅ No large intermediate datasets stored
- ✅ Minimal disk space usage
