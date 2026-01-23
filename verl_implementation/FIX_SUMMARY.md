# Self-Play Pipeline Fix Summary

**Date**: January 23, 2026  
**Status**: ✅ Implementation Complete

## Problem Diagnosis

The original self-play system had **4 critical bugs** preventing proper training:

### 1. Ground Truth Field Incorrect ❌
```python
# BEFORE (verl_implementation/data/preprocess_medec.py)
"reward_model": {
    "ground_truth": example.get("note_id", "")  # ❌ Using ID instead of label
}
```
**Impact**: Model saw IDs like "ms-train-656" as ground truth instead of "CORRECT"/"INCORRECT"

### 2. Placeholder Prompts Never Replaced ❌
```python
# BEFORE
initial_prompt = [
    {"role": "system", "content": "You are participating in a medical note game."},
    {"role": "user", "content": "Awaiting game initialization..."}  # ❌ Placeholder
]
```
**Impact**: Model received generic placeholder → responded with confusion

### 3. No Two-Phase Game Orchestration ❌
**Before**: Data preprocessing expected runtime tool invocation, but tool was never configured  
**Impact**: Single-turn classification only, no Injector→Assessor flow

### 4. Hardcoded Reward Function ❌
```python
# BEFORE (verl_implementation/reward/reward_function.py)
def compute_score(...):
    return 0.5  # ❌ Hardcoded placeholder
```
**Impact**: No learning signal, model couldn't improve

---

## Solution Implementation

### ✅ Fix 1: Proper Ground Truth Assignment

**File**: `verl_implementation/data/preprocess_selfplay.py`

```python
# NEW: Generate 2 examples per pair
def create_benign_example(pair, ...):
    return {
        "reward_model": {
            "ground_truth": "CORRECT"  # ✅ Actual label
        },
        "interaction_kwargs": {
            "name": "medical_game",
            "ground_truth": "CORRECT",
            "mode": "benign"
        }
    }

def create_error_example(pair, ...):
    return {
        "reward_model": {
            "ground_truth": "INCORRECT"  # ✅ Actual label
        },
        "interaction_kwargs": {
            "ground_truth": "INCORRECT",
            "mode": "error_injection"
        }
    }
```

**Result**: 405 pairs → 810 examples (405 CORRECT + 405 INCORRECT)

### ✅ Fix 2: Load Real Prompts from JSON

**File**: `verl_implementation/data/preprocess_selfplay.py`

```python
# NEW: Load actual prompts at preprocessing time
injection_prompts = load_prompts("configs/prompts/error_injection_prompts_v2.json")

# Benign mode
system_prompt = injection_prompts["system_prompt_correct"]
user_template = injection_prompts["injector_correct_template"]

# Error mode  
system_prompt = injection_prompts["system_prompt_incorrect"]
user_template = injection_prompts["injector_incorrect_template"]
```

**Result**: Model receives specific medical task instructions, not placeholders

### ✅ Fix 3: Implement Two-Phase Interaction

**File**: `scripts/self_play/interactions/medical_game_interaction.py`

```python
class MedicalGameInteraction(BaseInteraction):
    async def generate_response(self, instance_id, messages, **kwargs):
        if instance["turn"] == 1:
            # Phase 1: Process Injector output
            generated_note = self._extract_generated_note(injector_output)  # Strip <think>
            assessor_prompt = self._construct_assessor_prompt(generated_note)
            return False, assessor_prompt, 0.0, {}  # Continue to Phase 2
            
        elif instance["turn"] == 2:
            # Phase 2: Score Assessor classification
            assessor_answer = self._parse_assessor_answer(assessor_output)
            reward = 1.0 if assessor_answer == ground_truth else -1.0
            reward += 0.2 if has_valid_format else 0.0  # Format bonus
            return True, feedback, reward, {}  # Terminate
```

**Config**: `verl_implementation/config/interaction_config.yaml`

```yaml
interaction:
  - name: "medical_game"
    class_name: "scripts.self_play.interactions.medical_game_interaction.MedicalGameInteraction"
    config:
      detection_prompts_path: "configs/prompts/error_detection_prompts.json"
```

**Result**: Proper 2-turn game with CoT stripping between phases

### ✅ Fix 4: Zero-Sum Reward Calculation

**Implementation**: Built into `MedicalGameInteraction.generate_response()`

```python
# Assessor's classification vs ground_truth
if assessor_answer == ground_truth:
    reward = +1.0 + format_bonus  # Assessor wins
else:
    reward = -1.0 + format_bonus  # Injector wins (Assessor loses)
```

**Result**: Competitive self-improvement through zero-sum rewards

---

## Data Requirements Clarification

| Mode | Input Needed | ground_truth |
|------|--------------|--------------|
| **Benign** | Only `correct_note` | `"CORRECT"` |
| **Error Injection** | Full pair: `correct_note` + `incorrect_note` + `error_type` | `"INCORRECT"` |

**Rationale**: Error injection needs the incorrect example to understand what type of error to inject, but benign modifications only need the correct note.

---

## Training Flow

```
Data (810 examples)
    ↓
verl Loads Example with interaction_kwargs
    ↓
SGLang Rollout Begins
    ↓
┌─────────────────────────────────────────┐
│ TURN 1: INJECTOR                        │
│ Model generates modified note           │
│ Output includes <think> reasoning       │
└─────────────────────────────────────────┘
    ↓
MedicalGameInteraction.generate_response()
    ↓
    • Strip <think>...</think> tags
    • Extract generated_note:
    • Load Assessor prompt from error_detection_prompts.json
    • Return assessor_prompt (turn=2, continue)
    ↓
┌─────────────────────────────────────────┐
│ TURN 2: ASSESSOR                        │
│ Model sees ONLY modified note           │
│ Classifies as CORRECT or INCORRECT      │
└─────────────────────────────────────────┘
    ↓
MedicalGameInteraction.generate_response()
    ↓
    • Parse final_answer: "CORRECT" or "INCORRECT"
    • Compare to ground_truth
    • Compute reward: +1.0 (win) or -1.0 (lose)
    • Add format bonus: +0.2 if valid format
    • Return (terminate=True, reward)
    ↓
verl PPO Update
    ↓
Repeat for next example
```

---

## Verification Commands

```bash
# 1. Generate data
python verl_implementation/data/preprocess_selfplay.py

# 2. Verify parquet structure
python verl_implementation/scripts/verify_data.py

# 3. Test interaction logic
python verl_implementation/scripts/test_interaction.py

# 4. Start training
bash verl_implementation/scripts/run_training.sh
```

---

## Key Files Created

1. **`scripts/self_play/interactions/medical_game_interaction.py`** (408 lines)
   - Two-phase game orchestration
   - CoT stripping between phases
   - Zero-sum reward calculation

2. **`verl_implementation/data/preprocess_selfplay.py`** (189 lines)
   - Generates 810 examples from 405 pairs
   - Loads prompts from JSON configs
   - Sets proper ground_truth labels

3. **`verl_implementation/config/interaction_config.yaml`** (10 lines)
   - Registers MedicalGameInteraction with verl
   - Specifies detection prompts path

4. **`verl_implementation/scripts/run_training.sh`** (48 lines)
   - Launches verl training with multi-turn enabled
   - Configures interaction_config_path

5. **`verl_implementation/scripts/test_interaction.py`** (284 lines)
   - Unit tests for all game scenarios
   - Validates reward calculations

6. **`verl_implementation/scripts/verify_data.py`** (97 lines)
   - Checks parquet file structure
   - Validates interaction_kwargs format

---

## What Changed vs Original Design

### Original Plan (Not Implemented)
- Use verl's "tool" system with `BaseTool`
- Runtime prompt construction via tool invocation
- Separate tool_config.yaml

### Final Implementation (What We Built)
- Use verl's "interaction" system with `BaseInteraction`
- Pre-generate prompts at data preprocessing time
- Runtime only handles turn orchestration + reward

**Why**: verl's interaction system is specifically designed for multi-turn RL training with dynamic feedback, while tools are for external API calls. Interactions provide better integration with the rollout system.

---

## Results

✅ **Ground truth**: Now properly set to "CORRECT" / "INCORRECT"  
✅ **Prompts**: Loaded from JSON configs with specific medical instructions  
✅ **Two-phase game**: Injector → Assessor with CoT stripping  
✅ **Rewards**: Zero-sum (+1/-1) with format bonus (+0.2)  
✅ **Data**: 810 examples generated (729 train, 81 test)  
✅ **Tests**: All unit tests passing

**Ready for training!** 🚀
