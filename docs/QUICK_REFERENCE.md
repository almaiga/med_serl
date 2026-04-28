# ⚡ Quick Reference: Training Monitoring Commands

This is your cheat sheet for monitoring MedSeRL training (inspired by Hugging Face GRPO blog).

## 🎯 Before You Start Training

```bash
# 1. CRITICAL: Authenticate WandB (must do BEFORE training!)
wandb login
# → Visit shown URL, copy API key, paste when prompted

# 2. Install GPU monitor
pip install nvitop

# 3. Verify data exists
python verl_implementation/scripts/verify_data.py

# 4. (Optional) Run quick interaction test
python verl_implementation/scripts/test_interaction.py
```

## 🚀 During Training (Multi-Terminal Setup)

### Terminal 1: Start Training
```bash
# Option A: Original script
bash verl_implementation/scripts/run_training.sh

# Option B: Enhanced script with better logging
bash verl_implementation/scripts/run_training_monitored.sh
```

### Terminal 2: GPU Monitoring (Real-time)
```bash
# Shows VRAM, GPU util, temperature in real-time
nvitop

# What to watch:
# ✅ VRAM: 70-90% peak (not 100%, not <50%)
# ✅ GPU Util: 80-100% (spiky during generation is normal)
# ⚠️ Temp: <80°C (OK), >85°C (throttling risk)
```

### Terminal 3: View Logs
```bash
# Follow the console output
tail -f outputs/verl_training/*/log.txt

# Or use the real-time monitor
python verl_implementation/scripts/monitor_realtime.py
```

## 📊 After Training (Analysis)

### Quick Metrics Summary
```bash
# Analyze WandB run (fetches latest by default)
python verl_implementation/scripts/analyze_training.py

# Analyze specific run
python verl_implementation/scripts/analyze_training.py --run-id <run_id>

# Export metrics to CSV
python verl_implementation/scripts/analyze_training.py --export-csv metrics.csv
```

### View Interactive Dashboard
```bash
# Open in browser:
https://wandb.ai/home

# Or from terminal:
open https://wandb.ai/home  # macOS
# or
xdg-open https://wandb.ai/home  # Linux
```

## 🔍 What the Metrics Mean

| Metric | Good Range | Bad Signs |
|--------|-----------|-----------|
| **Reward Mean** | ↗️ 0.5 → 0.9 | 🔴 Flat/Declining |
| **KL Divergence** | 📈 Rising but <0.01 | 🔴 Flat (no learning) or >0.1 (drift) |
| **Gradient Norm** | 📉 Drops then stable | 🔴 Spikes or constantly climbing |
| **VRAM Usage** | 70-90% | 🔴 >95% (OOM risk) or <50% (underused) |
| **GPU Util** | 80-100% | 🔴 <50% (communication bottleneck) |
| **Loss** | 📉 Decreasing | 🔴 Increasing or NaN |

## ⚠️ Emergency: Training Crashes

### Out of Memory (OOM)
```bash
# Reduce batch size in run_training.sh:
TRAIN_BATCH_SIZE=256  # Was 512
# Then restart training
bash verl_implementation/scripts/run_training_monitored.sh
```

### Disk Full (Mid-Training)
```bash
# 1. Find corrupted checkpoint
ls -la outputs/verl_training/medserl-selfplay/*/global_step_*

# 2. Delete corrupted dir
rm -rf outputs/verl_training/medserl-selfplay/injector-assessor-game/global_step_XXX

# 3. Free space (delete old checkpoints)
rm -rf outputs/verl_training/medserl-selfplay/*/global_step_20
rm -rf outputs/verl_training/medserl-selfplay/*/global_step_40

# 4. Add resume flag to run_training.sh:
# trainer.resume_mode="auto" \

# 5. Restart
bash verl_implementation/scripts/run_training_monitored.sh
```

### WandB Not Logging
```bash
# Check authentication
cat ~/.netrc | grep wandb

# If not found, login again
wandb login

# When running, verify logging
python verl_implementation/scripts/analyze_training.py
```

## 📈 Performance Tuning (From GRPO Blog)

If training is **slow**:
```bash
# Check GPU util in nvitop

if [ LOW_GPU_UTIL ]; then
  # Reduce model sharding
  TENSOR_PARALLEL_SIZE=1  # Was higher
  ROLLOUT_GPU_MEMORY_UTIL=0.85  # Increase from 0.8
  TRAIN_BATCH_SIZE=1024  # Increase from 512
fi

if [ GPU_UTIL_SPIKY ]; then
  # Model is over-sharded
  TENSOR_PARALLEL_SIZE=1  # Switch to data parallelism
fi

if [ VERY_HIGH_VRAM ]; then
  # Close to OOM
  TRAIN_BATCH_SIZE=256  # Reduce from 512
  ROLLOUT_GPU_MEMORY_UTIL=0.7  # Reduce from 0.8
fi
```

## 🎯 Key Files

| File | Purpose |
|------|---------|
| `verl_implementation/scripts/run_training_monitored.sh` | Enhanced training launcher |
| `verl_implementation/scripts/analyze_training.py` | Post-training metric analysis |
| `verl_implementation/scripts/monitor_realtime.py` | Real-time terminal dashboard |
| `MONITORING_SETUP.md` | Detailed setup guide |
| `verl_implementation/config/ppo_trainer.yaml` | VERL training config |
| `verl_implementation/config/interaction_config.yaml` | Two-phase game config |

## 📚 Further Reading

- **GRPO Blog**: https://huggingface.co/blog/Weyaxi/engineering-handbook-grpo-lora-with-verl
- **VERL Docs**: https://verl.readthedocs.io/
- **WandB Guide**: https://docs.wandb.ai/guides/runs
- **nvitop GitHub**: https://github.com/XuehaiPan/nvitop

## 💡 Pro Tips

1. **Always log in to WandB before training** (saves 5+ mins of GPU time)
2. **Monitor first 20 minutes** (catches OOM/config issues early)  
3. **Check nvitop for GPU patterns**, not just raw numbers
4. **Set `trainer.save_total_limit=3`** in training script
5. **Test on 1 epoch first** to ensure setup works
6. **Save model checkpoints to Hugging Face** before deleting local ones
7. **Screenshot good metrics** from WandB dashboard for reports

## 🚨 Health Check Workflow

```bash
# 1. Start training (Terminal 1)
bash verl_implementation/scripts/run_training_monitored.sh

# 2. Monitor GPU immediately (Terminal 2)
nvitop
# Watch for: VRAM 70-90%, GPU >80%

# 3. After 2 minutes, check WandB
python verl_implementation/scripts/analyze_training.py
# Should show initial data points

# 4. After 20 minutes
# - Reward should show upward trend
# - KL div should be bounded
# - Gradient norm should be stable
# If all good, you can leave it running!
```

---

**Last Updated**: March 2026  
**Based on**: Hugging Face GRPO Engineering Handbook  
**Project**: MedSeRL Self-Play RL Training
