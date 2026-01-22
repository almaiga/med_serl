# MedSeRL verl Implementation

This directory contains the implementation of **MedSeRL** (Medical Self-play Error detection via Reinforcement Learning) using the **verl** framework with **REINFORCE++** algorithm.

## 📁 Structure

```
verl_implementation/
├── IMPLEMENTATION_PLAN.md      # Detailed implementation plan
├── README.md                   # This file
├── data/
│   └── preprocess_medec.py     # Convert MEDEC to verl format
├── reward/
│   └── medec_reward.py         # Custom reward function
└── scripts/
    ├── setup_verl.sh           # Environment setup
    ├── run_medserl_reinforce.sh # REINFORCE++ training
    └── run_medserl_grpo.sh     # GRPO training (alternative)
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install verl with vLLM backend
bash scripts/setup_verl.sh --backend vllm
```

### 2. Preprocess MEDEC Data

```bash
# Convert MEDEC data to verl parquet format
python data/preprocess_medec.py \
    --input_dir ../data_raw/MEDEC \
    --output_dir ~/data/medec
```

### 3. Run Training

```bash
# Train with REINFORCE++
bash scripts/run_medserl_reinforce.sh \
    --model Qwen/Qwen2.5-3B-Instruct \
    --gpus 1

# Or train with GRPO (often more stable)
bash scripts/run_medserl_grpo.sh \
    --model Qwen/Qwen2.5-3B-Instruct \
    --gpus 1
```

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
