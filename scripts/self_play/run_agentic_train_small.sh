#!/bin/bash
# MedSeRL Agentic Self-Play Training — SMALL-BATCH REAL RUN
#
# Two-pod setup via RunPod Global Networking:
#   Pod A (judge)    — Qwen3-8B vLLM server already running
#   Pod B (training) — this script, uses all local GPUs for the actor
#
# Usage:
#   export JUDGE_VLLM_URL=http://<judge-pod-id>.runpod.internal:8002/v1/chat/completions
#   bash scripts/self_play/run_agentic_train_small.sh
#
# Env overrides:
#   JUDGE_VLLM_URL  — REQUIRED: URL to the remote judge pod
#   ACTOR_MODEL     — Actor model path (default: Qwen/Qwen3-4B)
#   JUDGE_MODEL     — Judge model name for logging (default: Qwen/Qwen3-8B)
#   N_GPUS          — Number of local GPUs for training (default: 1)
#   MAX_PAIRS       — MEDEC pairs to generate (default: 100)
#   EPOCHS          — Training epochs (default: 5)
#   ROLLOUT_MAX_MODEL_LEN        — vLLM max model len (default: 8192)
#   ROLLOUT_MAX_BATCHED_TOKENS   — vLLM max batched tokens (default: 8192)
#   ROLLOUT_MAX_NUM_SEQS         — vLLM max concurrent sequences (default: 8)
#   ROLLOUT_RESPONSE_LENGTH      — max response length (default: 6144)
#   Agentic veRL uses async rollout; this script pins that mode explicitly.
#   ROLLOUT_ENFORCE_EAGER        — vLLM eager mode flag (default: True)
#   ROLLOUT_FREE_CACHE_ENGINE    — offload KV cache after rollout generation
#                                  (default: True, matching verl docs)
#   VLLM_GPU_MEM_UTIL            — vLLM GPU memory utilization (default: 0.7)
#   PPO_MICRO_BATCH_SIZE_PER_GPU — PPO micro-batch size per GPU (default: 2)
#   LOGPROB_MICRO_BATCH_SIZE_PER_GPU — rollout/ref log-prob micro-batch size per GPU (default: 2)
#   REWARD_NUM_WORKERS — parallel reward workers (default: 4)
#   REWARD_BACKEND    — `agentic` or `rule` (default: agentic)
#   UMLS_WEIGHT      — additive UMLS reward weight (default: 0.4; set 0 to disable)
#   UMLS_MAX_RPS     — per-process NLM request cap; defaults to a conservative
#                      share of the 20 req/s/IP limit across reward workers
#   VAL_MAX_SAMPLES  — validation examples per test pass (default: 16)
#   TEST_FREQ        — run validation every N steps (default: 5)
#   SAVE_FREQ       — save checkpoint every N steps (default: 25)
#   KEEP_ONLY_FINAL_CHECKPOINT — delete older global_step_* dirs at the end (default: 1)
#   SKIP_DATAGEN    — Set to 1 to reuse existing train.parquet

# ─── Configuration ────────────────────────────────────────────────────────────
ACTOR_MODEL="${ACTOR_MODEL:-Qwen/Qwen3-4B}"
export JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-8B}"
N_GPUS="${N_GPUS:-1}"

MAX_PAIRS="${MAX_PAIRS:-100}"
EPOCHS="${EPOCHS:-5}"
TRAIN_BATCH_SIZE=$(( N_GPUS * 16 ))          # 16 per GPU
TRAIN_SAMPLES=$(( EPOCHS * MAX_PAIRS ))      # 500 by default
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-8192}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-8192}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-8}"
ROLLOUT_RESPONSE_LENGTH="${ROLLOUT_RESPONSE_LENGTH:-6144}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-True}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.7}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-2}"
REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-4}"
REWARD_BACKEND="${REWARD_BACKEND:-agentic}"
UMLS_WEIGHT="${UMLS_WEIGHT:-0.4}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-16}"
TEST_FREQ="${TEST_FREQ:-5}"
SAVE_FREQ="${SAVE_FREQ:-25}"
KEEP_ONLY_FINAL_CHECKPOINT="${KEEP_ONLY_FINAL_CHECKPOINT:-1}"

# Paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_DIR="$PROJECT_ROOT/scripts/self_play/configs"
TRAIN_PARQUET="$PROJECT_ROOT/data_processed/self_play/train.parquet"
VAL_PARQUET="$PROJECT_ROOT/data_processed/self_play/val.parquet"

if [ -z "${UMLS_MAX_RPS:-}" ]; then
    UMLS_MAX_RPS=$(( 16 / REWARD_NUM_WORKERS ))
    if [ "$UMLS_MAX_RPS" -lt 1 ]; then
        UMLS_MAX_RPS=1
    fi
fi
export UMLS_MAX_RPS
export UMLS_WEIGHT

case "${REWARD_BACKEND,,}" in
    rule)
        CUSTOM_REWARD_PATH="$PROJECT_ROOT/scripts/self_play/reward_function.py"
        CUSTOM_REWARD_NAME="compute_score"
        ;;
    ""|agentic)
        REWARD_BACKEND="agentic"
        CUSTOM_REWARD_PATH="$PROJECT_ROOT/scripts/self_play/agentic_reward.py"
        CUSTOM_REWARD_NAME="async_compute_score"
        ;;
    *)
        echo "WARNING: Unknown REWARD_BACKEND='$REWARD_BACKEND'; using agentic."
        REWARD_BACKEND="agentic"
        CUSTOM_REWARD_PATH="$PROJECT_ROOT/scripts/self_play/agentic_reward.py"
        CUSTOM_REWARD_NAME="async_compute_score"
        ;;
esac

# Judge URL — must be set by the user
if [ -z "$JUDGE_VLLM_URL" ]; then
    echo "ERROR: JUDGE_VLLM_URL is not set."
    echo "  Set it to your judge pod's RunPod internal URL, e.g.:"
    echo "    export JUDGE_VLLM_URL=http://<pod-id>.runpod.internal:8002/v1/chat/completions"
    exit 1
fi
export JUDGE_VLLM_URL
export UMLS_API_KEY="${UMLS_API_KEY:-6878e795-ad79-4743-9758-546cacb8b31c}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

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
export RAY_USAGE_STATS_ENABLED=0
export RAY_enable_open_telemetry=0
export HYDRA_FULL_ERROR=1

# ── Disable OTLP/OpenTelemetry export inside Ray workers ──
# The training pod has hit intermittent SIGSEGVs in Ray background workers
# inside gRPC/OpenTelemetry metric export. Force local-only execution.
export OTEL_SDK_DISABLED=true
export OTEL_TRACES_EXPORTER=none
export OTEL_METRICS_EXPORTER=none
export OTEL_LOGS_EXPORTER=none
unset OTEL_EXPORTER_OTLP_ENDPOINT
unset OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
unset OTEL_EXPORTER_OTLP_METRICS_ENDPOINT
unset OTEL_EXPORTER_OTLP_LOGS_ENDPOINT
unset OTEL_EXPORTER_OTLP_HEADERS
unset OTEL_EXPORTER_OTLP_TRACES_HEADERS
unset OTEL_EXPORTER_OTLP_METRICS_HEADERS
unset OTEL_EXPORTER_OTLP_LOGS_HEADERS

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="small_agentic_${TIMESTAMP}"
OUTPUT_DIR="outputs/self_play/small_agentic_${TIMESTAMP}"

TRAIN_LOG="$OUTPUT_DIR/train.log"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_ROOT/results/self_play"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# ─── Cleanup: kill leftover Ray / GPU processes ───────────────────────────────
echo "=== Cleanup: killing stale Ray / GPU processes ==="
if pgrep -f "ray::" >/dev/null 2>&1; then
    echo "  Stopping Ray..."
    ray stop --force 2>/dev/null || true
    pkill -9 -f "ray::" 2>/dev/null || true
    sleep 2
fi
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -f "vllm.worker" 2>/dev/null || true

if command -v nvidia-smi &>/dev/null; then
    GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u)
    if [ -n "$GPU_PIDS" ]; then
        echo "  Found GPU processes: $GPU_PIDS — killing them..."
        for pid in $GPU_PIDS; do kill -9 "$pid" 2>/dev/null || true; done
        sleep 3
    else
        echo "  No stale GPU processes."
    fi
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
fi
echo "  Cleanup done."

# ─── Expand /dev/shm if too small ─────────────────────────────────────────────
if [ -d /dev/shm ]; then
    SHM_SIZE_KB=$(df /dev/shm 2>/dev/null | awk 'NR==2{print $2}')
    if [ -n "$SHM_SIZE_KB" ] && [ "$SHM_SIZE_KB" -lt 1048576 ]; then
        echo "  /dev/shm is only $((SHM_SIZE_KB/1024)) MB — expanding to 16 GB ..."
        mount -o remount,size=16G /dev/shm 2>/dev/null && echo "  /dev/shm expanded." \
            || echo "  WARNING: could not remount /dev/shm."
    fi
fi

# ─── Trap ─────────────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Trap: stopping Ray..."
    ray stop --force 2>/dev/null || true
    echo "  Done."
}
trap cleanup EXIT

echo "=================================================="
echo "MedSeRL Agentic Training — SMALL-BATCH REAL RUN"
echo "=================================================="
echo "Project root : $PROJECT_ROOT"
echo "Actor model  : $ACTOR_MODEL"
echo "Judge URL    : $JUDGE_VLLM_URL"
echo "N GPUs       : $N_GPUS"
echo "Max pairs    : $MAX_PAIRS"
echo "Epochs       : $EPOCHS"
echo "Train samples: $TRAIN_SAMPLES  (batch=$TRAIN_BATCH_SIZE)"
echo "Rollout len  : $ROLLOUT_RESPONSE_LENGTH"
echo "Rollout mode : async"
echo "Eager mode   : $ROLLOUT_ENFORCE_EAGER"
echo "Free cache   : $ROLLOUT_FREE_CACHE_ENGINE"
echo "Max model len: $ROLLOUT_MAX_MODEL_LEN"
echo "Max batched  : $ROLLOUT_MAX_BATCHED_TOKENS"
echo "Max seqs     : $ROLLOUT_MAX_NUM_SEQS"
echo "vLLM mem util: $VLLM_GPU_MEM_UTIL"
echo "Reward back. : $REWARD_BACKEND ($CUSTOM_REWARD_NAME)"
echo "Reward work. : $REWARD_NUM_WORKERS"
echo "UMLS weight  : $UMLS_WEIGHT"
echo "UMLS max RPS : $UMLS_MAX_RPS per worker"
echo "Save freq    : $SAVE_FREQ"
echo "Output dir   : $OUTPUT_DIR"
echo "=================================================="

# ─── Pre-flight: GLIBC patch ──────────────────────────────────────────────────
echo ""
echo "=== Pre-flight: Patching veRL engine imports (GLIBC workaround) ==="
python3 << 'PATCH_EOF'
import pathlib, re
fpath = pathlib.Path("/workspace/verl/verl/workers/engine/__init__.py")
if not fpath.exists():
    print("  SKIP: file not found")
else:
    code = fpath.read_text()
    new_code, n = re.subn(r'except ImportError:', 'except (ImportError, OSError):', code)
    if n == 0:
        print("  Already patched.")
    else:
        fpath.write_text(new_code)
        print(f"  PATCHED: widened {n} ImportError → (ImportError, OSError)")
PATCH_EOF

# ─── Pre-flight: Fix veRL fsdp_workers.py backend string bug ─────────────────
# veRL constructs "cpu:gloo,cpu:nccl" but PyTorch requires "cpu:gloo,cuda:nccl".
echo ""
echo "=== Pre-flight: Patching fsdp_workers.py (cpu:nccl → cuda:nccl) ==="
python3 << 'PATCH_FSDP'
import pathlib, re

for candidate in [
    "/workspace/verl/verl/workers/fsdp_workers.py",
    "/sgl-workspace/sglang/verl/verl/workers/fsdp_workers.py",
]:
    fpath = pathlib.Path(candidate)
    if not fpath.exists():
        continue
    code = fpath.read_text()
    # Fix any literal or f-string that produces "cpu:nccl"
    new_code, n = re.subn(r'cpu:nccl', 'cuda:nccl', code)
    if n == 0:
        print(f"  {fpath.name}: no 'cpu:nccl' found (already correct or different pattern).")
    else:
        fpath.write_text(new_code)
        print(f"  PATCHED {fpath}: replaced {n} occurrence(s) of 'cpu:nccl' → 'cuda:nccl'")
    break
else:
    print("  SKIP: fsdp_workers.py not found at known paths.")
PATCH_FSDP

# ─── Pre-flight: GPU memory check ────────────────────────────────────────────
echo ""
echo "=== Pre-flight: GPU memory check ==="
python3 << 'GPU_CHECK_EOF'
import subprocess, sys
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10
    )
    ok = True
    for line in result.stdout.strip().splitlines():
        idx, free_mb, total_mb = [int(x.strip()) for x in line.split(",")]
        free_gb, total_gb = free_mb / 1024, total_mb / 1024
        print(f"  GPU {idx}: {free_gb:.1f} GiB free / {total_gb:.1f} GiB total")
        if free_gb < 10:
            print(f"  WARNING: GPU {idx} has only {free_gb:.1f} GiB free.")
            ok = False
    if not ok:
        sys.exit(1)
    else:
        print("  OK — all GPUs have enough memory.")
except Exception as e:
    print(f"  WARNING: Could not check GPU memory: {e}")
GPU_CHECK_EOF
[ $? -ne 0 ] && { echo "ABORTING: Not enough GPU memory."; exit 1; }

# ─── Pre-flight: Judge connectivity check ────────────────────────────────────
# Normalize the URL: strip any path and reconstruct /v1/models endpoint
JUDGE_BASE_URL=$(echo "$JUDGE_VLLM_URL" | sed 's|/v1/.*||; s|/$||')
JUDGE_MODELS_URL="${JUDGE_BASE_URL}/v1/models"
# Also normalise completions URL in case caller set a bare base or wrong suffix
export JUDGE_VLLM_URL="${JUDGE_BASE_URL}/v1/chat/completions"

echo ""
echo "=== Pre-flight: Judge connectivity ($JUDGE_MODELS_URL) ==="
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 "$JUDGE_MODELS_URL" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    echo "  Judge reachable — HTTP $HTTP_CODE OK"
    echo "  JUDGE_VLLM_URL normalised to: $JUDGE_VLLM_URL"
else
    echo "  ERROR: Judge returned HTTP $HTTP_CODE (expected 200)."
    echo "  Tried: $JUDGE_MODELS_URL"
    echo ""
    echo "  Check that:"
    echo "    1. The judge pod is running: bash scripts/self_play/start_judge_server.sh"
    echo "    2. Both pods have RunPod Global Networking enabled"
    echo "    3. JUDGE_VLLM_URL format: http://<pod-id>.runpod.internal:8002/v1/chat/completions"
    echo "       Got: $JUDGE_VLLM_URL"
    exit 1
fi

# Quick sanity-check: one inference
echo "  Judge sanity check..."
curl -s "${JUDGE_VLLM_URL}" \
    -H "Content-Type: application/json" \
    -d '{"model":"'"${JUDGE_MODEL}"'","messages":[{"role":"user","content":"Reply OK."}],"max_tokens":5,"chat_template_kwargs":{"enable_thinking":false}}' \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print('  Judge OK:', r['choices'][0]['message']['content'])" \
    || echo "  WARNING: Judge sanity check failed — reward will fall back to rule-based."

# ─── Step 0: Data generation ──────────────────────────────────────────────────
echo ""
echo "=== Step 0: Data Generation (MAX_PAIRS=$MAX_PAIRS) ==="

if [ "${SKIP_DATAGEN:-0}" = "1" ] && [ -f "$TRAIN_PARQUET" ]; then
    echo "SKIP_DATAGEN=1 — reusing existing $TRAIN_PARQUET"
else
    echo "Generating train.parquet with $MAX_PAIRS pairs ..."
    python3 "$PROJECT_ROOT/scripts/self_play/preprocess_medec.py" \
        --input "$PROJECT_ROOT/data_processed/medec_paired/train_val_split/rl_train.jsonl" \
        --output "$TRAIN_PARQUET" \
        --injection-prompts "$PROJECT_ROOT/configs/prompts/error_injection_prompts_v4.json" \
        --max-pairs "$MAX_PAIRS"
    [ $? -ne 0 ] && { echo "ERROR: preprocess_medec.py failed. Aborting."; exit 1; }
    echo "train.parquet generated ($MAX_PAIRS pairs)."
fi

if [ ! -f "$VAL_PARQUET" ]; then
    echo "Copying train.parquet → val.parquet"
    cp "$TRAIN_PARQUET" "$VAL_PARQUET"
else
    echo "val.parquet found: $VAL_PARQUET"
fi

# ─── Step 1: Launch veRL REINFORCE++ training ─────────────────────────────────
echo ""
echo "=== Step 1: veRL REINFORCE++ Training ==="
echo "  Dataset   : $MAX_PAIRS pairs → $TRAIN_SAMPLES samples over $EPOCHS epochs"
echo "  Batch     : $TRAIN_BATCH_SIZE  |  Steps/epoch: $(( MAX_PAIRS / TRAIN_BATCH_SIZE ))"
echo "  Total steps: $(( TRAIN_SAMPLES / TRAIN_BATCH_SIZE ))"
echo "  Reward    : async_compute_score → $JUDGE_VLLM_URL"
echo ""

# Clean stale Ray state
echo "Cleaning stale Ray state..."
ray stop --force 2>/dev/null || true
rm -rf "$RAY_TMPDIR_PATH"/* /dev/shm/ray /tmp/ray 2>/dev/null || true
sleep 2

# Patch veRL Ray init for Docker
python3 << 'PATCH_RAY'
import pathlib, re
for candidate in [
    "/workspace/verl/verl/trainer/main_ppo.py",
    "/sgl-workspace/sglang/verl/verl/trainer/main_ppo.py",
]:
    fpath = pathlib.Path(candidate)
    if not fpath.exists():
        continue
    code = fpath.read_text()
    CANONICAL = """_ray_kw = OmegaConf.to_container(ray_init_kwargs)
    # ── MedSeRL Docker fix (canonical) ──
    import os as _os
    if _ray_kw.get('num_cpus') is None:
        _ray_kw['num_cpus'] = min(_os.cpu_count() or 4, 8)
    _ray_kw.setdefault('include_dashboard', False)
    _ray_kw.setdefault('_temp_dir', '/workspace/ray_tmp')
    _ray_kw.setdefault('_node_ip_address', '127.0.0.1')
    _ray_kw.setdefault('object_store_memory', 500_000_000)
    _ray_kw.setdefault('_plasma_directory', '/workspace/ray_tmp')
    print(f"ray init kwargs (patched): {_ray_kw}")
    ray.init(**_ray_kw)"""
    fresh = "ray.init(**OmegaConf.to_container(ray_init_kwargs))"
    if fresh in code:
        code = code.replace(fresh, CANONICAL)
        fpath.write_text(code)
        print(f"PATCHED {fpath}: fresh → canonical Docker-safe Ray init")
    elif "_ray_kw" in code:
        pattern = r'_ray_kw = OmegaConf\.to_container.*?ray\.init\(\*\*_ray_kw\)'
        new_code, n = re.subn(pattern, CANONICAL, code, count=1, flags=re.DOTALL)
        if n > 0:
            fpath.write_text(new_code)
            print(f"PATCHED {fpath}: replaced existing _ray_kw block")
        else:
            print(f"WARNING: {fpath} has _ray_kw but regex did not match")
    else:
        print(f"WARNING: could not find ray.init call in {fpath}")
    break
else:
    print("SKIP: main_ppo.py not found")
PATCH_RAY

# ─── WandB Setup ──────────────────────────────────────────────────────────────
echo ""
echo "=== WandB Setup ==="
if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY" --relogin 2>&1 | tail -1
    echo "  WandB: logged in via WANDB_API_KEY"
elif wandb status 2>/dev/null | grep -q "Logged in"; then
    echo "  WandB: already logged in"
else
    echo "  WARNING: WANDB_API_KEY not set and not logged in — logging to console only."
fi

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_DIR" \
    --config-name="ppo_agentic" \
    \
    algorithm.adv_estimator=reinforce_plus_plus \
    \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=16 \
    data.train_max_samples=$TRAIN_SAMPLES \
    data.val_max_samples=$VAL_MAX_SAMPLES \
    data.max_prompt_length=2048 \
    data.max_response_length=6144 \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path="$ACTOR_MODEL" \
    "++actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.85 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=$VLLM_GPU_MEM_UTIL \
    actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_BATCHED_TOKENS \
    actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS \
    actor_rollout_ref.rollout.enforce_eager=$ROLLOUT_ENFORCE_EAGER \
    actor_rollout_ref.rollout.free_cache_engine=$ROLLOUT_FREE_CACHE_ENGINE \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.prompt_length=2048 \
    actor_rollout_ref.rollout.response_length=$ROLLOUT_RESPONSE_LENGTH \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOGPROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPUS \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOGPROB_MICRO_BATCH_SIZE_PER_GPU \
    \
    critic.enable=false \
    \
    reward_model.enable=False \
    reward.num_workers=$REWARD_NUM_WORKERS \
    \
    custom_reward_function.path="$CUSTOM_REWARD_PATH" \
    custom_reward_function.name=$CUSTOM_REWARD_NAME \
    \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    trainer.total_epochs=$EPOCHS \
    trainer.critic_warmup=0 \
    "trainer.logger=[console,wandb]" \
    trainer.project_name=medserl-agentic-small \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.val_before_train=False \
    ++ray_kwargs.ray_init.include_dashboard=False \
    ++ray_kwargs.ray_init.num_cpus=8 \
    "++ray_kwargs.ray_init._temp_dir=$RAY_TMPDIR_PATH" \
    ++ray_kwargs.ray_init.object_store_memory=500000000 \
    2>&1 | tee "$TRAIN_LOG"

TRAIN_EXIT=${PIPESTATUS[0]}

# ─── Keep only final checkpoint for this run ──────────────────────────────────
if [ "$KEEP_ONLY_FINAL_CHECKPOINT" = "1" ] && [ -d "$OUTPUT_DIR" ]; then
    mapfile -t STEP_DIRS < <(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'global_step_*' | sort -V)
    if [ "${#STEP_DIRS[@]}" -gt 1 ]; then
        LAST_STEP_DIR="${STEP_DIRS[-1]}"
        echo ""
        echo "=== Checkpoint Cleanup ==="
        echo "Keeping final checkpoint: $LAST_STEP_DIR"
        for step_dir in "${STEP_DIRS[@]}"; do
            if [ "$step_dir" != "$LAST_STEP_DIR" ]; then
                echo "Removing older checkpoint: $step_dir"
                rm -rf "$step_dir"
            fi
        done
    fi
fi

# ─── Verification ─────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "=== Verification ==="
echo "=================================================="

CS_CALLS=$(grep -c "compute_score called" "$TRAIN_LOG" 2>/dev/null || true); CS_CALLS=${CS_CALLS:-0}
JUDGE_CALLS=$(grep -c "judge_verdict\|umls_score\|PASS\|FAIL\|ABSTAIN" "$TRAIN_LOG" 2>/dev/null || true); JUDGE_CALLS=${JUDGE_CALLS:-0}
SCORE_LINE=$(grep "critic/score/mean:" "$TRAIN_LOG" 2>/dev/null | tail -1 || true)

echo ""
echo "compute_score invocations : $CS_CALLS"
echo "Judge verdict entries     : $JUDGE_CALLS"
echo "Last critic/score/mean    : ${SCORE_LINE:-(not found)}"
echo "Training exit code        : $TRAIN_EXIT"
echo ""

if [ "$TRAIN_EXIT" -eq 0 ] && [ "$CS_CALLS" -gt 0 ]; then
    echo "TRAINING COMPLETED: agentic rewards were computed."
    echo ""
    echo "Recent score lines:"
    grep "critic/score/mean:" "$TRAIN_LOG" | tail -5
elif [ "$TRAIN_EXIT" -eq 0 ]; then
    echo "TRAINING COMPLETED (check reward routing — no compute_score log lines found)."
else
    echo "TRAINING FAILED: exit code $TRAIN_EXIT"
    echo "Check: $TRAIN_LOG"
fi

echo ""
echo "Full log : $TRAIN_LOG"
echo "Outputs  : $OUTPUT_DIR"

# ─── Response Quality Analysis ────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "=== Response Quality Analysis ==="
echo "=================================================="

python3 "$PROJECT_ROOT/scripts/self_play/analyze_smoke_quality.py" \
    --project-root        "$PROJECT_ROOT" \
    --output-dir          "$OUTPUT_DIR" \
    --smoke-log           "$TRAIN_LOG" \
    --max-response-length 4096

exit $TRAIN_EXIT
