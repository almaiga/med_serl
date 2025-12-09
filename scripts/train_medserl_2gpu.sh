#!/bin/bash
# MedSeRL Training Script for 2x A100 80GB GPUs
# Mirrors SeRL: Direct RL (REINFORCE++) from pretrained model, no SFT
#
# Usage:
#   python scripts/prepare_medec_for_openrlhf.py  # First, prepare data
#   bash scripts/train_medserl_2gpu.sh            # Then train
#   QUICK_TEST=1 bash scripts/train_medserl_2gpu.sh  # Quick test

set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Model - instruction-tuned (no SFT, like SeRL)
MODEL_PATH="${MODEL_PATH:-google/medgemma-4b-it}"

# Data paths (YOUR project, not SeRL)
PROMPT_DATA="${PROMPT_DATA:-$PROJECT_ROOT/data_processed/medec/seed_500.jsonl}"
EVAL_DATASET="${EVAL_DATASET:-$PROJECT_ROOT/data_processed/medec/test.jsonl}"

# Output paths
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${RUN_NAME:-medserl_$TIMESTAMP}"
SAVE_PATH="${SAVE_PATH:-$PROJECT_ROOT/outputs/$RUN_NAME}"
CKPT_PATH="${CKPT_PATH:-$SAVE_PATH/checkpoints}"

mkdir -p "$SAVE_PATH" "$CKPT_PATH"
mkdir -p "$SAVE_PATH/eval_outputs" "$SAVE_PATH/train_samples"
mkdir -p "$SAVE_PATH/filtered_data" "$SAVE_PATH/evolution_data"

# Reward functions (YOUR project)
REWARD_MAJ="$PROJECT_ROOT/src/training/medec_maj_reward.py"
REWARD_RULE="$PROJECT_ROOT/src/training/medec_reward.py"
FEW_SHOT_PROMPT="$PROJECT_ROOT/configs/prompts/medical_note_generation.json"

# SeRL hyperparameters
ACTOR_LR="${ACTOR_LR:-5e-7}"
KL_COEF="${KL_COEF:-1e-4}"
TRAIN_BATCH="${TRAIN_BATCH:-16}"
ROLLOUT_BATCH="${ROLLOUT_BATCH:-16}"
N_SAMPLES="${N_SAMPLES:-16}"
NUM_EPISODES="${NUM_EPISODES:-1000000}"
INSTR_PER_ITER="${INSTR_PER_ITER:-2000}"
DIFF_LOWER="${DIFF_LOWER:-0.2}"
DIFF_UPPER="${DIFF_UPPER:-0.8}"

# 2x A100 80GB config
NUM_GPUS=2
VLLM_ENGINES=2
VLLM_TP=1
VLLM_MEM=0.6

# Quick test mode
if [ "${QUICK_TEST:-0}" = "1" ]; then
    echo "=== QUICK TEST MODE ==="
    NUM_EPISODES=100
    INSTR_PER_ITER=100
    TRAIN_BATCH=8
    ROLLOUT_BATCH=8
    N_SAMPLES=8
fi

echo "=== MedSeRL Training (2x A100) ==="
echo "Model: $MODEL_PATH"
echo "Data: $PROMPT_DATA"
echo "Save: $SAVE_PATH"

# Start Ray if needed
if ! ray status &>/dev/null; then
    echo "Starting Ray..."
    ray start --head --node-ip-address 0.0.0.0 --num-gpus $NUM_GPUS
    sleep 5
fi

start_time=$(date +%s)

# Use SeRL's OpenRLHF (reference install)
OPENRLHF_DIR="$PROJECT_ROOT/SeRL/openrlhf"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="{\"working_dir\": \"$OPENRLHF_DIR\"}" \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node $NUM_GPUS \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node $NUM_GPUS \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node $NUM_GPUS \
    --vllm_num_engines $VLLM_ENGINES \
    --vllm_tensor_parallel_size $VLLM_TP \
    --vllm_gpu_memory_utilization $VLLM_MEM \
    --colocate_all_models \
    --advantage_estimator reinforce \
    --pretrain "$MODEL_PATH" \
    --remote_rm_url "$REWARD_MAJ,$REWARD_RULE" \
    --save_path "$SAVE_PATH/final" \
    --ckpt_path "$CKPT_PATH" \
    --save_hf_ckpt \
    --micro_train_batch_size 1 \
    --train_batch_size $TRAIN_BATCH \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size $ROLLOUT_BATCH \
    --n_samples_per_prompt $N_SAMPLES \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --max_samples 100000 \
    --generate_max_len 1024 \
    --zero_stage 3 \
    --save_steps 50 \
    --eval_steps 1 \
    --bf16 \
    --actor_learning_rate $ACTOR_LR \
    --init_kl_coef $KL_COEF \
    --prompt_data "$PROMPT_DATA" \
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
    --eval_output_root_dir "$SAVE_PATH/eval_outputs" \
    --train_samples_root_dir "$SAVE_PATH/train_samples" \
    --filtered_data_root_dir "$SAVE_PATH/filtered_data" \
    --eval_dataset "$EVAL_DATASET" \
    --self_reward_method bon_maj \
    --eval_n_samples_per_prompt 1 \
    --eval_temperature 0 \
    --reward_difficulty_bounds $DIFF_LOWER $DIFF_UPPER \
    --enable_self_evolution \
    --few_shot_generation 8 \
    --evolution_generation_data_root_dir "$SAVE_PATH/evolution_data" \
    --few_shot_generation_prompt "$FEW_SHOT_PROMPT" \
    --few_shot_generation_batch_size 4 \
    --num_episodes $NUM_EPISODES \
    --instructions_num_per_iteration $INSTR_PER_ITER \
    --wandb_run_name "$RUN_NAME"

end_time=$(date +%s)
echo "=== Done in $((end_time - start_time))s ==="
echo "Model: $SAVE_PATH/final"
