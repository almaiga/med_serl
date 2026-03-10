#!/bin/bash
# MedSeRL SGLang Multi-Turn Smoke Test
#
# Follows the official veRL sglang multiturn examples exactly:
#   https://github.com/verl-project/verl/tree/main/examples/sglang_multiturn
#   → run_qwen3-4b_gsm8k_multiturn.sh
#
# Usage:
#   bash scripts/self_play/run_rule_smoke.sh
ray stop
ray start --head --num-cpus=8 --num-gpus=1 --temp-dir=/dev/shm/ray --include-dashboard=false --disable-usage-stats --port=6379 --object-store-memory=10000000000
set -x
ulimit -n 65535

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="$PROJECT_DIR/scripts/self_play/configs"

ACTOR_MODEL="${ACTOR_MODEL:-Qwen/Qwen3-4B}"
N_GPUS="${N_GPUS:-1}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"

TRAIN_PARQUET="$PROJECT_DIR/data_processed/self_play/train.parquet"
VAL_PARQUET="$PROJECT_DIR/data_processed/self_play/val.parquet"

# One-time fix: undo any previous MedSeRL patches to main_ppo.py
python3 -c "
import pathlib, re
f = pathlib.Path('/workspace/verl/verl/trainer/main_ppo.py')
if f.exists() and '_ray_kw' in f.read_text():
    code = re.sub(r'_ray_kw = OmegaConf\.to_container.*?ray\.init\(\*\*_ray_kw\)',
                  'ray.init(**OmegaConf.to_container(ray_init_kwargs))',
                  f.read_text(), flags=re.DOTALL)
    f.write_text(code)
    print('Restored vanilla ray.init in main_ppo.py')
" 2>/dev/null || true

# Val file fallback
if [ ! -f "$VAL_PARQUET" ] && [ -f "$TRAIN_PARQUET" ]; then
    cp "$TRAIN_PARQUET" "$VAL_PARQUET"
fi

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name="ppo_sglang_smoke" \
    \
    algorithm.adv_estimator=reinforce_plus_plus \
    algorithm.use_kl_in_reward=False \
    \
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
    \
    actor_rollout_ref.model.path="$ACTOR_MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=2 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    critic.enable=false \
    reward_model.enable=False \
    \
    custom_reward_function.path="$PROJECT_DIR/scripts/self_play/reward_function.py" \
    custom_reward_function.name=compute_score \
    \
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
    "$@"
