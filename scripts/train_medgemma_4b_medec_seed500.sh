#!/bin/bash
# Training script for MedGemma-4B on MEDEC error detection (500 seed samples)
# Using Reinforce++ with majority voting reward (no self-evolution for initial test)

set -x

# Start time
start_time=$(date +%s)

# ==================================================================
# PATHS - UPDATE THESE FOR YOUR SERVER
# ==================================================================
PROJECT_ROOT="/workspace/med_serl"
SERL_ROOT="${PROJECT_ROOT}/SeRL"
WORKING_DIR="${SERL_ROOT}/openrlhf"

# Model paths
PRETRAIN_MODEL="/workspace/models/medgemma-4b-it"  # Update if different

# Data paths
SEED_DATA="${PROJECT_ROOT}/data_processed/medec/seed_500.jsonl"
TEST_DATA="${PROJECT_ROOT}/data_processed/medec/test.jsonl"

# Reward functions (existing medical reward functions - binary classification)
REWARD_MAJ="${SERL_ROOT}/openrlhf/reward_utils/med_maj_reward.py"
REWARD_GT="${SERL_ROOT}/openrlhf/reward_utils/med_reward.py"

# Output paths
SAVE_PATH="${PROJECT_ROOT}/outputs/medgemma_4b_medec_rl_seed500"
CKPT_PATH="${SAVE_PATH}/checkpoints"
EVAL_OUTPUT="${WORKING_DIR}/train_eval_outputs_dir"
FILTERED_DATA="${WORKING_DIR}/train_online_filtered_data_dir"

# Logging (set WANDB_KEY to your key or remove --use_wandb flag below)
WANDB_KEY=""  # Add your W&B key here if using
RUN_NAME="medgemma4b-medec-maj-seed500-$(date +%Y%m%d_%H%M%S)"

# ==================================================================
# HARDWARE CONFIGURATION (2x A100 80GB)
# ==================================================================
# Optimized for 2x A100 80GB GPUs
REF_NUM_GPUS=1
REWARD_NUM_GPUS=1
ACTOR_NUM_GPUS=2
VLLM_NUM_ENGINES=2
VLLM_TENSOR_PARALLEL=1

# ==================================================================
# TRAINING HYPERPARAMETERS
# ==================================================================
MICRO_TRAIN_BATCH=1        # Reduce if OOM
TRAIN_BATCH=8              # Smaller for initial test
MICRO_ROLLOUT_BATCH=2      # Reduce if OOM
ROLLOUT_BATCH=8            # Smaller for initial test
N_SAMPLES=16               # Number of responses per prompt
MAX_EPOCHS=1
PROMPT_MAX_LEN=1024        # Medical notes can be long
GENERATE_MAX_LEN=512       # Responses should be shorter
ACTOR_LR=5e-7
INIT_KL_COEF=1e-4

# Reward filtering (important to prevent reward hacking)
REWARD_DIFFICULTY_MIN=0.2
REWARD_DIFFICULTY_MAX=0.8

# ==================================================================
# LAUNCH TRAINING
# ==================================================================
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{\"working_dir\": \"${WORKING_DIR}\", \"excludes\":[\"dataset/\", \"evolution_generation_data_dir\", \"train_eval_outputs_dir/\", \"train_online_filtered_data_dir/\", \"train_samples_dir/\", \"*.whl\", \"__pycache__/\", \"*.pyc\"]}" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node ${REF_NUM_GPUS} \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node ${REWARD_NUM_GPUS} \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node ${ACTOR_NUM_GPUS} \
   --vllm_num_engines ${VLLM_NUM_ENGINES} \
   --vllm_tensor_parallel_size ${VLLM_TENSOR_PARALLEL} \
   --vllm_gpu_memory_utilization 0.6 \
   --advantage_estimator reinforce \
   --pretrain ${PRETRAIN_MODEL} \
   --remote_rm_url ${REWARD_MAJ},${REWARD_GT} \
   --save_path ${SAVE_PATH} \
   --ckpt_path ${CKPT_PATH} \
   --save_hf_ckpt \
   --micro_train_batch_size ${MICRO_TRAIN_BATCH} \
   --train_batch_size ${TRAIN_BATCH} \
   --micro_rollout_batch_size ${MICRO_ROLLOUT_BATCH} \
   --rollout_batch_size ${ROLLOUT_BATCH} \
   --n_samples_per_prompt ${N_SAMPLES} \
   --max_epochs ${MAX_EPOCHS} \
   --prompt_max_len ${PROMPT_MAX_LEN} \
   --max_samples 500 \
   --generate_max_len ${GENERATE_MAX_LEN} \
   --zero_stage 3 \
   --save_steps 50 \
   --eval_steps 50 \
   --bf16 \
   --actor_learning_rate ${ACTOR_LR} \
   --critic_learning_rate 9e-6 \
   --init_kl_coef ${INIT_KL_COEF} \
   --prompt_data ${SEED_DATA} \
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
   --wandb_run_name ${RUN_NAME} \
   --eval_output_root_dir ${EVAL_OUTPUT} \
   --filtered_data_root_dir ${FILTERED_DATA} \
   --eval_dataset ${TEST_DATA} \
   --self_reward_method bon_maj \
   --eval_n_samples_per_prompt 1 \
   --eval_temperature 0 \
   --reward_difficulty_bounds ${REWARD_DIFFICULTY_MIN} ${REWARD_DIFFICULTY_MAX} \
   --num_episodes 1

   # Optional flags (uncomment if needed):
   # --apply_chat_template \
   # --colocate_all_models \  # Use if you have limited GPUs

# End time
end_time=$(date +%s)
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time seconds"
