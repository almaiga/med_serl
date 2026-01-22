# MedSeRL Implementation Plan with verl + REINFORCE++

## Overview

This document outlines the implementation of **MedSeRL** (Medical Error Detection via Reinforcement Learning) using the **verl** framework with **REINFORCE++** algorithm.

**Primary Goal**: Train a model to **detect medical errors** in clinical notes through **medical reasoning**, not pattern matching.

## 🎯 What This Implementation Does

### Task: Medical Error Detection (Binary Classification with Reasoning)
Given a clinical note, the model must:
1. **Analyze** the clinical information using medical knowledge
2. **Detect** whether there is a medical error (ERROR or CORRECT)
3. **Explain** the reasoning behind the decision

### Why Reinforcement Learning?
Medical error detection requires **clinical reasoning** that goes beyond surface-level pattern matching:
- Understanding drug-disease interactions
- Recognizing inappropriate treatments for specific conditions
- Identifying diagnostic inconsistencies with clinical findings
- Applying medical guidelines and standard of care

RL allows the model to learn nuanced medical decision-making through reward signals.

## 🎯 Key Technical Details

### Algorithm: REINFORCE++
verl supports REINFORCE++ directly via `algorithm.adv_estimator=reinforce_plus_plus`:
```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus \
    ...
```

---

## 📁 Project Structure

```
verl_implementation/
├── IMPLEMENTATION_PLAN.md          # This file
├── data/
│   └── preprocess_medec.py         # Convert preprocessed JSONL to verl parquet
├── reward/
│   └── medec_reward.py             # Reward function for error detection
├── scripts/
│   ├── run_medserl_reinforce.sh    # Training launch script
│   ├── run_medserl_grpo.sh         # Alternative with GRPO
│   └── setup_verl.sh               # Environment setup
└── README.md
```

---

## 📊 Data: Already Preprocessed!

Your MEDEC data is already preprocessed at:
```
data_processed/medec_paired/train_val_split/
├── rl_train.jsonl    (405 examples)
└── sft_train.jsonl   (1213 examples)
```

### Data Format (JSONL)
Each line contains:
```json
{
    "note_id": "ms-train-1591",
    "incorrect_note": "<clinical note with medical error>",
    "correct_note": "<corrected clinical note>",
    "error_type": "management|diagnosis|treatment|pharmacotherapy|causalorganism",
    "error_sentence": "<sentence containing the error>",
    "corrected_sentence": "<corrected sentence>"
}
```

### Error Types (Require Medical Reasoning)
| Type | Example | Reasoning Required |
|------|---------|--------------------|
| **diagnosis** | Fibroadenoma → Phyllodes tumor | Rapid growth + age + size inconsistent with benign fibroadenoma |
| **management** | Azithromycin → Penicillin | VDRL+/FTA-ABS+ = syphilis, first-line treatment is penicillin |
| **pharmacotherapy** | Interferon → Tenofovir | SLE patient on hydroxychloroquine - interferon contraindicated |
| **treatment** | Metoclopramide → Diet modification | High HbA1c diabetic gastroparesis needs glucose control first |
| **causalorganism** | E. coli → Norovirus | No travel, vegan diet, acute onset = viral, not bacterial |

---

## 🔄 Task: Medical Error Detection

### Input
A clinical note that may contain a medical error.

### Output (JSON)
```json
{
    "assessment": "ERROR",
    "reasoning": "<medical reasoning explaining WHY this is an error>",
    "error_type": "management"
}
```

### Key Insight: Medical Reasoning, Not Pattern Matching
The model cannot simply memorize "Azithromycin is wrong" - it must understand:
- Positive VDRL + FTA-ABS confirms syphilis diagnosis
- First-line treatment for syphilis is penicillin (per guidelines)
- Azithromycin + ceftriaxone is the gonorrhea/chlamydia regimen
- This is a **management error** - wrong treatment for the confirmed diagnosis

### verl Data Format
verl expects parquet files with:
```python
{
    "data_source": "medec",
    "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "<note>"}],
    "ability": "medical_error_detection",
    "reward_model": {"style": "rule", "ground_truth": {...}},
    "extra_info": {"note_id": "...", "split": "train"}
}
```

### Reward Function Design

The reward function evaluates **medical reasoning quality**, not just correctness:

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Reward function for medical error detection.
    
    Rewards:
        +1.0: Correct detection with valid medical reasoning
        +0.5: Correct detection, weak reasoning
        +0.3: Correct detection, no reasoning
         0.0: Invalid/unparseable response
        -0.5: False positive (hallucinated error)
        -0.8: False negative (missed real error) - CRITICAL in medical context
    
    The asymmetric penalty reflects clinical reality:
    missing a real error is more dangerous than a false alarm.
    """
```

### Why Reasoning Matters for Reward
- A model that says "ERROR" without reasoning may be pattern-matching
- A model that explains "Penicillin is first-line for syphilis per CDC guidelines" demonstrates actual medical knowledge
- Reasoning-based rewards encourage **generalizable medical understanding**

---

## 🚀 Quick Start

### Step 1: Install verl
```bash
bash verl_implementation/scripts/setup_verl.sh --backend vllm
```

### Step 2: Convert Preprocessed Data to verl Format
```bash
# Your data is already at data_processed/medec_paired/train_val_split/
python verl_implementation/data/preprocess_medec.py \
    --input_jsonl data_processed/medec_paired/train_val_split/rl_train.jsonl \
    --output_dir ~/data/medec
```

### Step 3: Run REINFORCE++ Training
```bash
bash verl_implementation/scripts/run_medserl_reinforce.sh \
    --model Qwen/Qwen2.5-3B-Instruct \
    --gpus 1
```

---

## 📊 Training Configuration

### Key Parameters for REINFORCE++
```yaml
algorithm:
  adv_estimator: reinforce_plus_plus  # or grpo
  
data:
  train_batch_size: 256
  max_prompt_length: 1024
  max_response_length: 256  # Detection + reasoning doesn't need long output

actor_rollout_ref:
  model:
    path: Qwen/Qwen2.5-3B-Instruct  # Start small
  actor:
    optim:
      lr: 1e-6
    ppo_mini_batch_size: 64
    use_kl_loss: True
    kl_loss_coef: 0.001
  rollout:
    n: 4  # Number of responses per prompt for REINFORCE++
```

---

## 🔬 Evaluation Metrics

### Primary Metrics
1. **Detection Accuracy**: Binary classification (ERROR vs CORRECT)
2. **False Negative Rate**: Critical - missing real errors is dangerous
3. **Error Type Accuracy**: Correct classification of error category

### Secondary Metrics
4. **Reasoning Quality**: Does the explanation demonstrate medical knowledge?
5. **Calibration**: Is the model confident when correct, uncertain when wrong?

---

## 📈 Expected Timeline

| Phase | Task | Duration |
|-------|------|----------|
| 1 | Convert data to verl format | 1 day |
| 2 | Basic REINFORCE++ training | 3-5 days |
| 3 | Hyperparameter tuning | 1 week |
| 4 | Evaluation on held-out test set | 1-2 days |

---

## ⚠️ Important Notes

### Hardware Requirements
- Minimum: 1x GPU with 24GB VRAM (RTX 3090/4090, A100)
- For 7B models: 2-4x GPUs recommended
- For 3B models: 1x GPU sufficient

### Why verl?
1. Native REINFORCE++ support
2. Clean integration with vLLM for fast inference
3. Easy custom reward functions
4. Active development and good documentation

### Key verl Files to Study
```
verl/
├── trainer/main_ppo.py          # Main entry point
├── utils/reward_score/          # Reward function examples
└── examples/
    └── reinforce_plus_plus_trainer/  # REINFORCE++ examples
```

---

## 💡 Medical Reasoning Examples

### Example 1: Management Error
**Note**: "VDRL and FTA-ABS positive. Azithromycin and ceftriaxone prescribed."

**Required Reasoning**:
- VDRL+/FTA-ABS+ = confirmed syphilis (not just suspected)
- Azithromycin + ceftriaxone = empiric treatment for gonorrhea/chlamydia
- First-line syphilis treatment = Penicillin G (per CDC STI guidelines)
- This is wrong treatment for confirmed diagnosis → **Management Error**

### Example 2: Diagnosis Error  
**Note**: "6cm rapidly growing breast mass. Diagnosed with fibroadenoma."

**Required Reasoning**:
- Fibroadenomas are typically <3cm and slow-growing
- Rapid growth + large size (6cm) + multinodular = concerning features
- Phyllodes tumor presents exactly this way
- Age 48 is atypical for new fibroadenoma
- This is incorrect diagnosis → **Diagnosis Error**

---

## 🔗 References

- [verl Documentation](https://verl.readthedocs.io/)
- [verl GitHub](https://github.com/volcengine/verl)
- [REINFORCE++ Examples](https://github.com/volcengine/verl/tree/main/examples/reinforce_plus_plus_trainer)
