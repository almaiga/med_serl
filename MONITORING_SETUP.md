# Training Monitoring Setup (GRPO-style)

This guide sets up comprehensive monitoring for your MedSeRL VERL training, following the patterns from the Hugging Face GRPO blog.

## 📊 Key Metrics to Monitor

Based on the GRPO engineering handbook, track:

1. **System Health** (GPU/VRAM Usage)
   - VRAM utilization percentage
   - GPU compute utilization
   - Temperature

2. **Training Progress**
   - Reward mean (how well model aligns with reward function)
   - KL divergence (policy drift from reference model)
   - Gradient norm (numerical stability)
   - Response length (token generation efficiency)

3. **Convergence Indicators**
   - Validation accuracy
   - Loss curves (actor, critic)
   - Learning rate schedule

## 🔐 Step 1: Setup WandB Authentication

**⚠️ CRITICAL:** Must do this BEFORE training to avoid crashes after 5 minutes of initialization!

```bash
# Login to WandB once (saves credential locally)
wandb login

# When prompted, visit https://wandb.ai/authorize and paste your API key
```

**Verify authentication:**
```bash
# Check if .netrc file exists
cat ~/.netrc | grep wandb
```

## 📈 Step 2: Install GPU Monitoring Tool

```bash
# Install nvitop for real-time GPU telemetry
pip install nvitop
```

## 🚀 Step 3: Run Training with Monitoring

### Terminal 1: Start Training
```bash
bash verl_implementation/scripts/run_training.sh
```

### Terminal 2: Monitor GPU in Real-Time
```bash
# Watch GPU/VRAM/Temperature in real-time (refreshes every 1 second)
nvitop
```

## 📊 Step 4: View Results in WandB Dashboard

1. Open: https://wandb.ai/home
2. Find project: `medserl-selfplay`
3. Find run: `injector-assessor-game-{timestamp}`
4. Create custom charts:
   - **VRAM Usage vs Training Step**
   - **GPU Utilization vs Training Step**
   - **Reward Mean vs Training Step**
   - **KL Divergence vs Training Step**
   - **Gradient Norm vs Training Step**

## 📋 Expected Metric Ranges

### Healthy Training Profile:
- **VRAM Usage**: 70-90% peak (not 100%, leaves room for gradients)
- **GPU Utilization**: 80-100% with slight dips during generation
- **Reward Mean**: Monotonic increase (starts ~0.5, climbs to ~0.9)
- **KL Divergence**: Steady rise but bounded (peak <0.01)
- **Gradient Norm**: Stable after initial 20 steps, no spikes

### Warning Signs:
- **VRAM Usage**: >95% → OOM risk, reduce batch size
- **Zero GPU Util**: Communication overhead, check `tensor_model_parallel_size`
- **Reward Flat**: Learning rate too low or algorithm issue
- **KL Divergence >0.1**: Model drifting too far, increase KL penalty
- **Gradient Spikes**: Exploding gradients, reduce learning rate

## 🔍 Debugging Slow Training

If training is slow, check these in `nvitop`:

1. **GPU Util low + High VRAM underuse**: 
   - Reduce `tensor_model_parallel_size` (switch to Data Parallelism)
   - Increase `rollout.gpu_memory_utilization`

2. **VRAM Saturation (100%) → OOM crash**:
   - Reduce `data.train_batch_size`
   - Reduce `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`

3. **GPU Util spiky/inconsistent**:
   - Model is over-sharded across GPUs
   - Try `tensor_model_parallel_size=1`

## 📁 Log Locations

- **WandB cloud**: logs/media automatically synced
- **Local checkpoint dir**: `outputs/verl_training/medserl-selfplay/injector-assessor-game/`
- **Training logs**: Check console output for epoch summaries

## 💡 Pro Tips

1. **Start with conservative batch sizes** (test training stability first)
2. **Monitor for first 20 minutes** (catches OOM/config issues early)
3. **Save checkpoints frequently** (set `trainer.save_freq=5` for recovery)
4. **Use `trainer.resume_mode="auto"`** if training interrupts
5. **Set `trainer.save_total_limit=3`** to manage disk space (auto-delete old checkpoints)

## 🛑 Emergency: Training Crash

If training crashes due to disk full (like in the blog):

```bash
# 1. Delete corrupted checkpoint
rm -rf outputs/verl_training/medserl-selfplay/injector-assessor-game/global_step_*

# 2. Free up space
rm -rf outputs/verl_training/medserl-selfplay/injector-assessor-game/global_step_20
rm -rf outputs/verl_training/medserl-selfplay/injector-assessor-game/global_step_40

# 3. Add resume flag to run_training.sh
# trainer.resume_mode="auto" \

# 4. Re-run training
bash verl_implementation/scripts/run_training.sh
```

## 📚 References

- Original blog: https://huggingface.co/blog/Weyaxi/engineering-handbook-grpo-lora-with-verl
- nvitop docs: https://github.com/XuehaiPan/nvitop
- WandB docs: https://docs.wandb.ai/
- VERL docs: https://verl.readthedocs.io/
