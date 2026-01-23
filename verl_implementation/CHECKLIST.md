# MedSeRL Self-Play Implementation Checklist

## ✅ Completed Tasks

### Phase 1: Diagnosis
- [x] Identified 4 critical bugs in original pipeline
- [x] Analyzed verl documentation for interaction system
- [x] Confirmed data requirements (benign needs only correct_note, error needs full pair)
- [x] Understood reward structure (zero-sum with format bonus)

### Phase 2: Data Pipeline
- [x] Created `preprocess_selfplay.py` to generate 810 examples from 405 pairs
- [x] Fixed ground_truth assignment (CORRECT/INCORRECT instead of note_id)
- [x] Loaded real prompts from JSON configs (not placeholders)
- [x] Generated balanced dataset (405 benign + 405 error)
- [x] Split into train (729) and test (81) parquet files

### Phase 3: Interaction System
- [x] Implemented `MedicalGameInteraction` class with BaseInteraction
- [x] Phase 1 (Injector): Parse output, strip CoT, build Assessor prompt
- [x] Phase 2 (Assessor): Parse classification, compute zero-sum reward
- [x] Format validation with +0.2 bonus for proper structure
- [x] Created `interaction_config.yaml` to register with verl

### Phase 4: Testing & Verification
- [x] Created `verify_data.py` to check parquet structure
- [x] Created `test_interaction.py` with 4 unit tests
- [x] All tests passing (benign correct, benign wrong, error correct, invalid format)
- [x] Verified label balance (50/50 CORRECT/INCORRECT)

### Phase 5: Training Infrastructure
- [x] Created `run_training.sh` with multi-turn configuration
- [x] Configured SGLang backend with interaction system
- [x] Set up W&B logging (project: medserl-selfplay)
- [x] Created `setup_selfplay.sh` for one-command setup

### Phase 6: Documentation
- [x] Updated `README.md` with complete pipeline overview
- [x] Created `FIX_SUMMARY.md` documenting all 4 bug fixes
- [x] Added inline code comments explaining key logic
- [x] Created this checklist

---

## 📊 Current Status

**Data**:
- ✅ 810 examples generated (729 train, 81 test)
- ✅ Prompts loaded from JSON configs
- ✅ Ground truth properly set
- ✅ Interaction kwargs formatted correctly

**Code**:
- ✅ MedicalGameInteraction implemented (408 lines)
- ✅ Two-phase game flow working
- ✅ CoT stripping functional
- ✅ Zero-sum rewards computed correctly

**Testing**:
- ✅ Data verification passing
- ✅ Interaction unit tests passing (4/4)
- ✅ Format bonus working (+0.2)

**Ready for Training**: ✅ YES

---

## 🚀 Next Steps

### Immediate (Before Training)
1. [ ] Review training hyperparameters in `run_training.sh`
   - Batch size: 512 (adjust based on GPU memory)
   - Learning rate: 5e-7 (may need tuning)
   - Epochs: 50 (monitor for convergence)

2. [ ] Set up W&B account/project
   ```bash
   wandb login
   # Project will be: medserl-selfplay
   # Experiment: injector-assessor-game
   ```

3. [ ] Verify GPU availability
   ```bash
   nvidia-smi
   # Need sufficient VRAM for google/medgemma-4b-it
   ```

### During Training
1. [ ] Monitor key metrics:
   - Assessor accuracy (should increase over time)
   - Format compliance rate (should be high)
   - Average reward per phase
   - PPO loss convergence

2. [ ] Check for issues:
   - [ ] "Awaiting game initialization" → interaction not enabled
   - [ ] All rewards = 0.5 → reward function not working
   - [ ] Model gibberish → prompts not loaded
   - [ ] OOM errors → reduce batch size

3. [ ] Save checkpoints regularly (every 5 epochs configured)

### After Training
1. [ ] Evaluate on test set
   ```bash
   # Run inference on test.parquet
   # Measure Assessor accuracy
   # Compare to baseline performance
   ```

2. [ ] Analyze failure modes:
   - Where does Assessor fail?
   - What types of errors are hardest to detect?
   - Is format bonus helping?

3. [ ] Scale up if successful:
   - Use larger training set (sft_train.jsonl)
   - Increase model size
   - Adjust difficulty (more subtle errors)

---

## 🐛 Known Limitations

1. **Fixed difficulty**: Error injection follows prompts exactly, no adaptive difficulty
   - **Future**: Adjust error subtlety based on Assessor performance

2. **Limited error types**: Only diagnosis, management, pharmacotherapy
   - **Future**: Add more error categories (dosage, laterality, timeline)

3. **No curriculum learning**: All examples presented equally
   - **Future**: Start with easier errors, gradually increase difficulty

4. **Binary classification**: Only CORRECT/INCORRECT, no error localization
   - **Future**: Add sentence-level error detection

5. **Small dataset**: 405 pairs → 810 examples
   - **Future**: Use full training set when stable (1000+ pairs)

---

## 📝 Configuration Summary

### Data
- **Source**: `data_processed/medec_paired/train_val_split/rl_train.jsonl`
- **Output**: `data_processed/selfplay/train.parquet` (729) + `test.parquet` (81)
- **Split**: 90% train, 10% test

### Model
- **Architecture**: google/medgemma-4b-it
- **Backend**: SGLang with vLLM
- **Multi-turn**: Enabled (2 turns max)

### Training
- **Algorithm**: PPO with GAE
- **Batch size**: 512
- **Learning rate**: 5e-7
- **Epochs**: 50
- **KL coefficient**: 0.001
- **Save frequency**: Every 5 epochs

### Interaction
- **System**: verl BaseInteraction
- **Phases**: 2 (Injector → Assessor)
- **Rewards**: Zero-sum (+1/-1) + format bonus (+0.2)
- **CoT**: Stripped between phases

### Prompts
- **Injector**: `configs/prompts/error_injection_prompts_v2.json`
- **Assessor**: `configs/prompts/error_detection_prompts.json`

---

## 🎯 Success Criteria

**Minimum Viable**:
- [ ] Training completes without errors
- [ ] Assessor accuracy > random (50%)
- [ ] Model follows output format consistently

**Good Performance**:
- [ ] Assessor accuracy > 70% on test set
- [ ] Format compliance > 90%
- [ ] Clear improvement over epochs

**Excellent Performance**:
- [ ] Assessor accuracy > 85% on test set
- [ ] Robust to adversarial Injector attempts
- [ ] Generalizes to unseen error types

---

## 📞 Troubleshooting Guide

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| "Awaiting game initialization" | Interaction not enabled | Check `multi_turn.enable=True` |
| Ground truth = IDs | Wrong preprocessing script | Use `preprocess_selfplay.py` |
| All rewards = 0.5 | Interaction not computing | Check `interaction_config.yaml` path |
| Model nonsense | Prompts not loaded | Verify JSON file paths |
| OOM | Batch size too large | Reduce to 256 or 128 |
| No convergence | Learning rate wrong | Try 1e-6 or 1e-5 |
| Low format compliance | Prompt unclear | Add more examples to prompts |

---

## 📚 File Reference

**Core Implementation**:
- `scripts/self_play/interactions/medical_game_interaction.py` - Game orchestration
- `verl_implementation/data/preprocess_selfplay.py` - Data generation
- `verl_implementation/config/interaction_config.yaml` - verl configuration

**Utilities**:
- `verl_implementation/scripts/verify_data.py` - Data validation
- `verl_implementation/scripts/test_interaction.py` - Unit tests
- `verl_implementation/scripts/setup_selfplay.sh` - One-command setup

**Training**:
- `verl_implementation/scripts/run_training.sh` - Main training script

**Documentation**:
- `verl_implementation/README.md` - Overview and quick start
- `verl_implementation/FIX_SUMMARY.md` - Detailed bug fixes
- This file - Complete checklist

---

## ✅ Final Pre-Flight Check

Before running `bash verl_implementation/scripts/run_training.sh`:

1. [ ] Data files exist:
   - [ ] `data_processed/selfplay/train.parquet` (729 examples)
   - [ ] `data_processed/selfplay/test.parquet` (81 examples)

2. [ ] Config files exist:
   - [ ] `configs/prompts/error_injection_prompts_v2.json`
   - [ ] `configs/prompts/error_detection_prompts.json`
   - [ ] `verl_implementation/config/interaction_config.yaml`

3. [ ] Tests passing:
   - [ ] `python verl_implementation/scripts/verify_data.py` ✓
   - [ ] `python verl_implementation/scripts/test_interaction.py` ✓

4. [ ] Environment ready:
   - [ ] verl installed
   - [ ] SGLang backend available
   - [ ] GPU visible (`nvidia-smi`)
   - [ ] W&B configured (optional but recommended)

5. [ ] Hyperparameters reviewed:
   - [ ] Batch size appropriate for GPU
   - [ ] Learning rate reasonable (5e-7 default)
   - [ ] Number of epochs set (50 default)

**If all checked**: 🚀 **READY TO LAUNCH!**

```bash
bash verl_implementation/scripts/run_training.sh
```

**Monitor progress in**:
- Console output (real-time logs)
- W&B dashboard (metrics/charts)
- Checkpoint directory (saved models)
