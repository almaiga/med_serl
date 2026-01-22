# MedSeRL Self-Play Training

Self-play RL training for medical error detection using verl framework.

## Overview

This implementation uses:
- **Framework**: verl (HybridFlow-based RL training)
- **Algorithm**: REINFORCE++ (variance-reduced policy gradient)
- **Model**: Qwen/Qwen3-4B (actor and critic)
- **Backend**: PyTorch FSDP

## Configuration Approach

Following verl's official patterns, we **use verl's default `ppo_trainer.yaml`** config and override parameters via command-line arguments. This is the recommended approach by the verl team and avoids configuration validation issues.

**Note**: Do not create custom YAML configs with `_target_` fields unless you're very familiar with Hydra's structured configs. All official verl examples use the default config + command-line overrides pattern.

## Files

- `run_training.sh`: Main training script
- `preprocess_medec.py`: Converts MEDEC JSONL to parquet format for verl
- `configs/self_play.yaml`: **DEPRECATED** - use command-line overrides instead

## Usage

```bash
cd /path/to/med_serl
bash scripts/self_play/run_training.sh
```

## Key Parameters

### Data
- `data.train_batch_size=64`: Training batch size
- `data.max_prompt_length=256`: Max input tokens
- `data.max_response_length=512`: Max output tokens

### Actor (Doctor Agent)
- `actor_rollout_ref.actor.ppo_mini_batch_size=16`: PPO mini-batch size
- `actor_rollout_ref.actor.optim.lr=1e-6`: Learning rate
- `actor_rollout_ref.actor.ppo_epochs=2`: PPO epochs per batch

### Critic (Scribe Agent)
- `critic.optim.lr=1e-5`: Critic learning rate  
- `critic.ppo_mini_batch_size=16`: Critic batch size

### Algorithm
- `algorithm.adv_estimator=reinforce_plus_plus`: REINFORCE++ advantage estimation
- `algorithm.gamma=1.0`: Discount factor
- `algorithm.lam=0.95`: GAE lambda parameter

### Training
- `trainer.total_epochs=3`: Number of training epochs
- `trainer.save_freq=500`: Save checkpoint every N steps
- `trainer.test_freq=100`: Run validation every N steps

## Debugging

If you encounter errors:

1. **Config validation errors**: Make sure you're using verl's default config, not a custom YAML
2. **CUDA OOM**: Reduce `train_batch_size`, `ppo_mini_batch_size`, or enable offloading
3. **vllm errors**: Check `gpu_memory_utilization` (try lowering to 0.3)

## References

- verl quickstart: https://github.com/volcengine/verl/blob/main/docs/start/quickstart.rst
- verl examples: https://github.com/volcengine/verl/tree/main/examples/ppo_trainer
