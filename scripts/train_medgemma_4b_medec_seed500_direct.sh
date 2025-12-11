#!/bin/bash
# Direct training script without Ray job submission
# This runs the training command directly

set -x
start_time=$(date +%s)

# Paths
PROJECT_ROOT="/workspace/med_serl"
SERL_ROOT="${PROJECT_ROOT}/SeRL"
WORKING_DIR="${SERL_ROOT}/openrlhf"

# Change to working directory
cd "${WORKING_DIR}"

# Run training directly
# GPU allocation for 2 GPUs total:
# - vLLM engines: share GPU 0 (0.6 utilization)
# - Ref model: share GPU 0
# - Reward model: share GPU 1  
# - Actor/Critic: use both GPUs with ZeRO-3
python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --vllm_gpu_memory_utilization 0.4 \
   --colocate_actor_ref \
   --advantage_estimator reinforce \
   --pretrain google/medgemma-4b-it \
   --remote_rm_url ${SERL_ROOT}/openrlhf/reward_utils/med_maj_reward.py,${SERL_ROOT}/openrlhf/reward_utils/med_reward.py \
   --save_path ${PROJECT_ROOT}/outputs/medgemma_4b_medec_rl_seed500 \
   --ckpt_path ${PROJECT_ROOT}/outputs/medgemma_4b_medec_rl_seed500/checkpoints \
   --save_hf_ckpt \
   --micro_train_batch_size 1 \
   --train_batch_size 8 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 8 \
   --n_samples_per_prompt 16 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 500 \
   --generate_max_len 512 \
   --zero_stage 3 \
   --save_steps 50 \
   --eval_steps 50 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 1e-4 \
   --prompt_data ${PROJECT_ROOT}/data_processed/medec/seed_500.jsonl \
   --input_key problem \
   --label_key answer \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --wandb_run_name medgemma4b-medec-direct-$(date +%Y%m%d_%H%M%S) \
   --eval_output_root_dir ${WORKING_DIR}/train_eval_outputs_dir \
   --train_samples_root_dir ${WORKING_DIR}/train_samples_dir \
   --filtered_data_root_dir ${WORKING_DIR}/train_online_filtered_data_dir \
   --eval_dataset ${PROJECT_ROOT}/data_processed/medec/test.jsonl \
   --self_reward_method bon_maj \
   --eval_n_samples_per_prompt 1 \
   --eval_temperature 0 \
   --reward_difficulty_bounds 0.2 0.8 \
   --num_episodes 1

end_time=$(date +%s)
execution_time=$((end_time - start_time))
echo "Training completed in $execution_time seconds"
