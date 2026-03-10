#!/bin/bash
# MedSeRL SGLang Multi-Turn Smoke Test (rule-based reward, no judge)
#
# Tests the full veRL REINFORCE++ loop with SGLang multi-turn rollouts
# and the rule-based reward_function.py — NO judge server needed.
#
# Usage:
#   bash scripts/self_play/run_rule_smoke.sh

set -x
ulimit -n 65535

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="$PROJECT_DIR/scripts/self_play/configs"

ACTOR_MODEL="${ACTOR_MODEL:-Qwen/Qwen3-4B}"
N_GPUS="${N_GPUS:-1}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"

TRAIN_PARQUET="$PROJECT_DIR/data_processed/self_play/train.parquet"
VAL_PARQUET="$PROJECT_DIR/data_processed/self_play/val.parquet"

# All Ray temp under /workspace (persistent, large, won't break SSH)
RAY_TMPDIR_PATH="/workspace/ray_tmp"
mkdir -p "$RAY_TMPDIR_PATH"

# ── Ray env vars for RunPod Docker ──
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_DEDUP_LOGS=0
export RAY_USE_MULTIPROCESSING_CPU_COUNT=1
export RAY_memory_monitor_refresh_ms=0
export RAY_raylet_start_wait_time_s=300
export RAY_TMPDIR="$RAY_TMPDIR_PATH"
export RAY_GCS_SERVER_REQUEST_TIMEOUT_S=60
export HYDRA_FULL_ERROR=1

# ── Gentle Ray cleanup (safe for SSH) ──
ray stop --force 2>/dev/null || true
rm -rf "$RAY_TMPDIR_PATH"/* 2>/dev/null || true
sleep 2

# ── Patch veRL Ray init for Docker ──
python3 << 'PATCH_RAY'
import pathlib
fpath = pathlib.Path("/workspace/verl/verl/trainer/main_ppo.py")
if not fpath.exists():
    print("SKIP: main_ppo.py not found")
else:
    code = fpath.read_text()
    target = "ray.init(**OmegaConf.to_container(ray_init_kwargs))"
    if "_ray_kw" not in code and target in code:
        replacement = """_ray_kw = OmegaConf.to_container(ray_init_kwargs)
    # ── MedSeRL Docker fix ──
    import os as _os
    if _ray_kw.get('num_cpus') is None:
        _ray_kw['num_cpus'] = min(_os.cpu_count() or 4, 8)
    _ray_kw.setdefault('include_dashboard', False)
    _ray_kw.setdefault('_temp_dir', '/workspace/ray_tmp')
    _ray_kw.setdefault('_node_ip_address', '127.0.0.1')
    print(f"ray init kwargs (patched): {_ray_kw}")
    ray.init(**_ray_kw)"""
        code = code.replace(target, replacement)
        fpath.write_text(code)
        print("PATCHED: main_ppo.py — Docker-safe Ray defaults")
    else:
        print("main_ppo.py already patched or target not found")
PATCH_RAY

# ── Val file fallback ──
if [ ! -f "$VAL_PARQUET" ] && [ -f "$TRAIN_PARQUET" ]; then
    cp "$TRAIN_PARQUET" "$VAL_PARQUET"
fi

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name="ppo_sglang_smoke" \
    algorithm.adv_estimator=reinforce_plus_plus \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    data.train_batch_size=8 \
    data.train_max_samples=20 \
    data.val_max_samples=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$ACTOR_MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=2 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.enable=false \
    reward_model.enable=False \
    custom_reward_function.path="$PROJECT_DIR/scripts/self_play/reward_function.py" \
    custom_reward_function.name=compute_score \
    trainer.total_epochs="$TRAIN_EPOCHS" \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=medserl-smoke \
    trainer.experiment_name=smoke_sglang \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=99999 \
    trainer.val_before_train=False \
    ++ray_kwargs.ray_init.include_dashboard=False \
    ++ray_kwargs.ray_init.num_cpus=8 \
    "++ray_kwargs.ray_init._temp_dir=$RAY_TMPDIR_PATH" \
    "$@"
