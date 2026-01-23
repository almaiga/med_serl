# MedSeRL Self-Play Training Pipeline

Two-phase self-play game for medical error detection: **Injector → Assessor**

## Overview

The self-play system trains a model to both generate and detect medical errors through a competitive game:

### Phase 1: Injector
- **Benign mode**: Makes minimal surface edits (1-3 words) while preserving clinical meaning
- **Error mode**: Injects subtle clinical errors (diagnosis, management, pharmacotherapy)
- **Output**: Modified clinical note (with hidden `<think>` reasoning stripped before Phase 2)

### Phase 2: Assessor
- **Input**: Only the modified note (no original note or Injector's reasoning)
- **Task**: Classify as `CORRECT` or `INCORRECT` with medical reasoning
- **Output**: `final_answer: "CORRECT"` or `"INCORRECT"` + explanation

### Rewards (Zero-Sum)
- **Assessor correct**: Assessor +1.0, Injector -1.0
- **Assessor wrong**: Assessor -1.0, Injector +1.0
- **Format bonus**: +0.2 for following required output format

## 📁 Structure

```
verl_implementation/
├── README.md                            # This file
├── config/
│   └── interaction_config.yaml          # Interaction system config
├── data/
│   ├── preprocess_medec.py             # Original MEDEC preprocessor
│   └── preprocess_selfplay.py          # Self-play data generator (NEW)
├── reward/
│   └── reward_function.py              # Reward computation
└── scripts/
    ├── run_training.sh                  # Main training launch script (NEW)
    ├── test_interaction.py              # Unit tests for game flow (NEW)
    └── verify_data.py                   # Validate parquet files (NEW)
```

## 🚀 Quick Start

### 1. Generate Self-Play Data

From 405 note pairs → 810 training examples (405 benign + 405 error)

```bash
python verl_implementation/data/preprocess_selfplay.py \
    --input data_processed/medec_paired/train_val_split/rl_train.jsonl \
    --output_dir data_processed/selfplay

# Output:
#   data_processed/selfplay/train.parquet (729 examples)
#   data_processed/selfplay/test.parquet (81 examples)
```

### 2. Verify Data

```bash
python verl_implementation/scripts/verify_data.py
```

Checks: parquet structure, required fields, label balance, interaction kwargs

### 3. Test Interaction System

```bash
python verl_implementation/scripts/test_interaction.py
```

Validates: two-phase game flow, CoT stripping, reward calculation, format bonus

### 4. Start Training

```bash
bash verl_implementation/scripts/run_training.sh
```

Training config:
- **Model**: `google/medgemma-4b-it`
- **Backend**: SGLang with multi-turn rollout
- **Interaction**: 2-phase (Injector → Assessor)
- **Batch size**: 512, **Epochs**: 50, **LR**: 5e-7

## 🎯 Task Description

The model learns to detect medical errors in clinical notes:

**Input**: A clinical note that may contain a medical error

**Output** (JSON):
```json
{
    "assessment": "ERROR",
    "reasoning": "The prescription is inappropriate for the condition.",
    "error_sentence": "The patient was prescribed aspirin for bacterial infection.",
    "corrected_sentence": "The patient was prescribed amoxicillin for bacterial infection.",
    "error_type": "pharmacotherapy"
}
```

**Error Types**:
- `diagnosis` - Incorrect diagnostic conclusions
- `management` - Wrong disease management approach
- `treatment` - Inappropriate treatment decisions
- `pharmacotherapy` - Medication errors
- `causalorganism` - Wrong pathogen identification

## 📊 Reward Function

The reward function (`reward/medec_reward.py`) evaluates:

| Component | Points | Description |
|-----------|--------|-------------|
| Detection | +0.4 | Correct error/no-error classification |
| Localization | +0.3 | Matching error sentence |
| Correction | +0.2 | Quality of correction |
| Error Type | +0.1 | Correct error classification |
| False Negative | -0.7 | Missing an actual error |
| False Positive | -0.5 | Hallucinating an error |

## ⚙️ Configuration Options

### REINFORCE++
```bash
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct  # Base model
NUM_GPUS=4                           # Number of GPUs
TRAIN_BATCH_SIZE=256                 # Batch size
ROLLOUT_N=4                          # Responses per prompt
LR=1e-6                              # Learning rate
```

### GRPO
```bash
GROUP_SIZE=8                         # Group size for GRPO
# GRPO uses relative reward comparison within groups
```

## 🔬 Algorithm Details

### REINFORCE++

REINFORCE++ is an enhanced version of the REINFORCE algorithm with:
- Per-token baseline for variance reduction
- KL penalty to prevent policy collapse
- Entropy bonus for exploration

verl configuration:
```yaml
algorithm:
  adv_estimator: reinforce_plus_plus
```

### GRPO (Alternative)

Group Relative Policy Optimization:
- Groups multiple responses per prompt
- Ranks within groups for relative advantage
- Often more stable than REINFORCE++

verl configuration:
```yaml
algorithm:
  adv_estimator: grpo
actor_rollout_ref:
  rollout:
    n: 8  # Group size
```

## 📈 Monitoring

Training logs are available via:
- **Console**: Real-time metrics
- **Weights & Biases**: Full experiment tracking

```bash
# Set your W&B API key
export WANDB_API_KEY=your_key_here
```

## 🔄 Future: Self-Play Extension

Phase 2 will add adversarial self-play where the model plays two roles:
- **Injector**: Creates errors or benign changes
- **Assessor**: Detects and corrects errors

See `IMPLEMENTATION_PLAN.md` for details.

## 🐛 Troubleshooting

### Out of Memory
- Reduce `TRAIN_BATCH_SIZE`
- Enable gradient checkpointing (default: enabled)
- Use a smaller model

### Slow Training
- Increase `gpu_memory_utilization` in rollout config
- Use more GPUs for tensor parallelism
- Reduce `max_response_length`

### Poor Convergence
- Lower learning rate
- Increase KL penalty (`kl_loss_coef`)
- Switch from REINFORCE++ to GRPO

## 📚 References

- [verl Documentation](https://verl.readthedocs.io/)
- [verl GitHub](https://github.com/volcengine/verl)
- [REINFORCE++ in verl](https://github.com/volcengine/verl/tree/main/examples/reinforce_plus_plus_trainer)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
