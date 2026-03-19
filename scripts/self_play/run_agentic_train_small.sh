#!/bin/bash
# MedSeRL Agentic Self-Play Training — SMALL-BATCH REAL RUN
#
# Real training run (not a smoke test) with live UMLS judge.
#
# GPU 0 → Qwen3-8B judge server (vLLM, port 8002)
# GPU 1 → Qwen3-4B actor training (veRL REINFORCE++, fsdp2)
#
# Rewards computed in real-time via agentic_reward.py (async_compute_score),
# using the 3-step UMLS pipeline: entity extraction → UMLS retrieval → adjudication.
#
# Usage:
#   chmod +x scripts/self_play/run_agentic_train_small.sh
#   bash scripts/self_play/run_agentic_train_small.sh
#
# Env overrides:
#   ACTOR_MODEL   — Actor model path (default: Qwen/Qwen3-4B)
#   JUDGE_MODEL   — Judge model path (default: Qwen/Qwen3-8B)
#   JUDGE_PORT    — vLLM judge port  (default: 8002)
#   JUDGE_GPU     — GPU index for judge (default: 0)
#   TRAIN_GPU     — GPU index for training (default: 1)
#   MAX_PAIRS     — MEDEC pairs to generate (default: 100)
#   EPOCHS        — Training epochs (default: 5)
#   SKIP_SERVER   — Set to 1 to reuse an already-running judge server
#   SKIP_DATAGEN  — Set to 1 to reuse existing train.parquet

# ─── Configuration ────────────────────────────────────────────────────────────
ACTOR_MODEL="${ACTOR_MODEL:-Qwen/Qwen3-4B}"
export JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-8B}"
JUDGE_PORT="${JUDGE_PORT:-8002}"
SKIP_SERVER="${SKIP_SERVER:-0}"
SKIP_DATAGEN="${SKIP_DATAGEN:-0}"

JUDGE_GPU="${JUDGE_GPU:-0}"
TRAIN_GPU="${TRAIN_GPU:-1}"

MAX_PAIRS="${MAX_PAIRS:-100}"
EPOCHS="${EPOCHS:-5}"
TRAIN_BATCH_SIZE=16
TRAIN_SAMPLES=$(( EPOCHS * MAX_PAIRS ))   # 500 by default

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

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="small_agentic_${TIMESTAMP}"
OUTPUT_DIR="outputs/self_play/small_agentic_${TIMESTAMP}"

# Paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_DIR="$PROJECT_ROOT/scripts/self_play/configs"
TRAIN_PARQUET="$PROJECT_ROOT/data_processed/self_play/train.parquet"
VAL_PARQUET="$PROJECT_ROOT/data_processed/self_play/val.parquet"
TRAIN_LOG="$OUTPUT_DIR/train.log"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_ROOT/results/self_play"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# ─── Cleanup: kill leftover processes from previous runs ─────────────────────
echo "=== Cleanup: killing ALL stale GPU / vLLM / Ray processes ==="
if lsof -ti :${JUDGE_PORT} >/dev/null 2>&1; then
    echo "  Killing processes on port ${JUDGE_PORT}..."
    lsof -ti :${JUDGE_PORT} | xargs kill -9 2>/dev/null || true
    sleep 2
fi
if pgrep -f "ray::" >/dev/null 2>&1; then
    echo "  Stopping Ray..."
    ray stop --force 2>/dev/null || true
    pkill -9 -f "ray::" 2>/dev/null || true
    sleep 2
fi
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -f "vllm.worker" 2>/dev/null || true
pkill -9 -f "from multiprocessing" 2>/dev/null || true

echo "  Checking GPU processes via nvidia-smi..."
if command -v nvidia-smi &>/dev/null; then
    GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u)
    if [ -n "$GPU_PIDS" ]; then
        echo "  Found GPU processes: $GPU_PIDS — killing them..."
        for pid in $GPU_PIDS; do
            kill -9 "$pid" 2>/dev/null || true
        done
        sleep 3
    else
        echo "  No stale GPU processes found."
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
            || echo "  WARNING: could not remount /dev/shm (not root?). Will cap object_store_memory instead."
    else
        echo "  /dev/shm is $((SHM_SIZE_KB/1024)) MB — OK."
    fi
fi

# ─── Pre-flight: GPU memory check ────────────────────────────────────────────
echo ""
echo "=== Pre-flight: GPU memory check (all GPUs) ==="
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
        free_gb = free_mb / 1024
        total_gb = total_mb / 1024
        print(f"  GPU {idx}: {free_gb:.1f} GiB free / {total_gb:.1f} GiB total")
        if free_gb < 10:
            print(f"  WARNING: GPU {idx} has only {free_gb:.1f} GiB free (need ≥10 GiB).")
            ok = False
    if not ok:
        print("  ERROR: one or more GPUs lack sufficient free memory.")
        sys.exit(1)
    else:
        print("  OK — all GPUs have enough memory.")
except Exception as e:
    print(f"  WARNING: Could not check GPU memory: {e}")
GPU_CHECK_EOF
if [ $? -ne 0 ]; then
    echo "ABORTING: Not enough GPU memory. Clean up stale processes first."
    exit 1
fi

# ─── Trap: ensure cleanup on exit ────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Trap: cleaning up all GPU processes..."
    if [ -n "$VLLM_PID" ]; then
        echo "  Stopping judge server (PID $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
    fi
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    pkill -9 -f "vllm.worker" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    if command -v nvidia-smi &>/dev/null; then
        GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u)
        for pid in $GPU_PIDS; do
            kill -9 "$pid" 2>/dev/null || true
        done
    fi
    echo "  Cleanup complete."
}
trap cleanup EXIT

echo "=================================================="
echo "MedSeRL Agentic Training — SMALL-BATCH REAL RUN"
echo "=================================================="
echo "Project root : $PROJECT_ROOT"
echo "Actor model  : $ACTOR_MODEL"
echo "Judge model  : $JUDGE_MODEL"
echo "Judge port   : $JUDGE_PORT"
echo "Judge GPU    : $JUDGE_GPU  |  Train GPU: $TRAIN_GPU"
echo "Max pairs    : $MAX_PAIRS"
echo "Epochs       : $EPOCHS"
echo "Train samples: $TRAIN_SAMPLES  (batch=$TRAIN_BATCH_SIZE)"
echo "Output dir   : $OUTPUT_DIR"
echo "=================================================="

# ─── Pre-flight: Patch veRL engine imports (GLIBC workaround) ────────────────
echo ""
echo "=== Pre-flight: Patching veRL engine imports (GLIBC workaround) ==="
python3 << 'PATCH_EOF'
import pathlib, re

fpath = pathlib.Path("/workspace/verl/verl/workers/engine/__init__.py")
if not fpath.exists():
    print("  SKIP: file not found")
else:
    code = fpath.read_text()
    new_code, n = re.subn(
        r'except ImportError:',
        'except (ImportError, OSError):',
        code,
    )
    if n == 0:
        print("  No bare 'except ImportError:' found (already patched?).")
    else:
        fpath.write_text(new_code)
        print(f"  PATCHED: widened {n} 'except ImportError' → 'except (ImportError, OSError)'")
PATCH_EOF

# ─── Step 0: Ensure data exists ───────────────────────────────────────────────
echo ""
echo "=== Step 0: Data Generation (MAX_PAIRS=$MAX_PAIRS) ==="

if [ "$SKIP_DATAGEN" = "1" ] && [ -f "$TRAIN_PARQUET" ]; then
    echo "SKIP_DATAGEN=1 — reusing existing $TRAIN_PARQUET"
elif [ -f "$TRAIN_PARQUET" ] && [ "$SKIP_DATAGEN" != "0" ]; then
    echo "train.parquet found — skipping generation. Set SKIP_DATAGEN=0 to regenerate."
else
    echo "Generating train.parquet with $MAX_PAIRS pairs ..."
    python3 "$PROJECT_ROOT/scripts/self_play/preprocess_medec.py" \
        --input "$PROJECT_ROOT/data_processed/medec_paired/train_val_split/rl_train.jsonl" \
        --output "$TRAIN_PARQUET" \
        --injection-prompts "$PROJECT_ROOT/configs/prompts/error_injection_prompts_v4.json" \
        --max-pairs "$MAX_PAIRS"
    if [ $? -ne 0 ]; then
        echo "ERROR: preprocess_medec.py failed. Aborting."
        exit 1
    fi
    echo "train.parquet generated ($MAX_PAIRS pairs)."
fi

if [ ! -f "$VAL_PARQUET" ]; then
    echo "val.parquet missing — copying train.parquet as val.parquet"
    cp "$TRAIN_PARQUET" "$VAL_PARQUET"
else
    echo "val.parquet found: $VAL_PARQUET"
fi

# ─── Step 1: Start judge server ───────────────────────────────────────────────
echo ""
echo "=== Step 1: Judge Server (${JUDGE_MODEL} on GPU ${JUDGE_GPU}) ==="

JUDGE_URL="http://localhost:${JUDGE_PORT}/v1/chat/completions"
export JUDGE_VLLM_URL="$JUDGE_URL"
export UMLS_API_KEY="${UMLS_API_KEY:-6878e795-ad79-4743-9758-546cacb8b31c}"

if [ "$SKIP_SERVER" = "1" ]; then
    echo "SKIP_SERVER=1 — assuming judge already running at $JUDGE_URL"
else
    if curl -s -o /dev/null -w "%{http_code}" "${JUDGE_URL%/chat/completions}/models" 2>/dev/null | grep -q "200"; then
        echo "Judge server already running at $JUDGE_URL — skipping launch."
    else
        echo "Pre-downloading judge model weights..."
        python3 -c "
from huggingface_hub import snapshot_download
import os
print('Downloading ${JUDGE_MODEL} ...')
snapshot_download('${JUDGE_MODEL}', ignore_patterns=['*.gguf'])
print('Download complete.')
" || echo "Pre-download skipped (model may already be cached)."

        echo "Starting vLLM judge server on GPU ${JUDGE_GPU}, port ${JUDGE_PORT} ..."
        CUDA_VISIBLE_DEVICES=$JUDGE_GPU python3 -m vllm.entrypoints.openai.api_server \
            --model "${JUDGE_MODEL}" \
            --port "${JUDGE_PORT}" \
            --dtype bfloat16 \
            --max-model-len 4096 \
            --gpu-memory-utilization 0.50 \
            --enforce-eager \
            --served-model-name "${JUDGE_MODEL}" \
            &
        VLLM_PID=$!
        echo "vLLM PID: $VLLM_PID"

        echo "Waiting for judge server to become healthy (may take 5-10 min)..."
        WAIT_SECS=0
        MAX_WAIT=900
        until curl -s -o /dev/null -w "%{http_code}" "http://localhost:${JUDGE_PORT}/health" 2>/dev/null | grep -q "200"; do
            sleep 10
            WAIT_SECS=$((WAIT_SECS + 10))
            echo -n "."
            if [ "$WAIT_SECS" -ge "$MAX_WAIT" ]; then
                echo ""
                echo "ERROR: Judge server did not start within ${MAX_WAIT}s. Aborting."
                kill "$VLLM_PID" 2>/dev/null || true
                exit 1
            fi
        done
        echo ""
        echo "Judge server healthy after ${WAIT_SECS}s at $JUDGE_URL"
    fi
fi

echo "Judge server sanity check..."
curl -s "${JUDGE_URL}" \
    -H "Content-Type: application/json" \
    -d '{"model":"'"${JUDGE_MODEL}"'","messages":[{"role":"user","content":"Reply OK."}],"max_tokens":5,"chat_template_kwargs":{"enable_thinking":false}}' \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print('Judge OK:', r['choices'][0]['message']['content'])" \
    || echo "WARNING: Judge sanity check failed — reward may fall back to rule-based."

# ─── Step 2: Launch veRL REINFORCE++ training ─────────────────────────────────
echo ""
echo "=== Step 2: veRL REINFORCE++ Training ==="
echo "  Dataset : $MAX_PAIRS pairs → $TRAIN_SAMPLES samples over $EPOCHS epochs"
echo "  Batch   : $TRAIN_BATCH_SIZE  |  Steps/epoch: $(( MAX_PAIRS / TRAIN_BATCH_SIZE ))"
echo "  Total steps: $(( TRAIN_SAMPLES / TRAIN_BATCH_SIZE ))"
echo "  Config  : ppo_agentic.yaml"
echo "  Reward  : async_compute_score (UMLS judge + rule-based hybrid)"
echo ""
echo "JUDGE_VLLM_URL=$JUDGE_VLLM_URL"
echo "UMLS_API_KEY set: $([ -n "$UMLS_API_KEY" ] && echo yes || echo NO)"
echo ""

# Clean stale Ray state
echo "Cleaning stale Ray state..."
ray stop --force 2>/dev/null || true
rm -rf "$RAY_TMPDIR_PATH"/* /dev/shm/ray /tmp/ray 2>/dev/null || true
sleep 2
echo "Ray state cleaned."

# Patch veRL Ray init for Docker
python3 << 'PATCH_RAY'
import pathlib, re
fpath = pathlib.Path("/workspace/verl/verl/trainer/main_ppo.py")
if not fpath.exists():
    print("SKIP: main_ppo.py not found")
else:
    code = fpath.read_text()

    CANONICAL = """_ray_kw = OmegaConf.to_container(ray_init_kwargs)
    # ── MedSeRL Docker fix (canonical) ──
    import os as _os
    if _ray_kw.get('num_cpus') is None:
        _ray_kw['num_cpus'] = min(_os.cpu_count() or 4, 8)
    _ray_kw.setdefault('include_dashboard', False)
    _ray_kw.setdefault('_temp_dir', '/workspace/ray_tmp')
    _ray_kw.setdefault('_node_ip_address', '127.0.0.1')
    _ray_kw.setdefault('object_store_memory', 500_000_000)   # 500 MB
    _ray_kw.setdefault('_plasma_directory', '/workspace/ray_tmp')  # bypass tiny /dev/shm
    print(f"ray init kwargs (patched): {_ray_kw}")
    ray.init(**_ray_kw)"""

    fresh = "ray.init(**OmegaConf.to_container(ray_init_kwargs))"
    if fresh in code:
        code = code.replace(fresh, CANONICAL)
        fpath.write_text(code)
        print("PATCHED: main_ppo.py — fresh → canonical Docker-safe Ray init")
    elif "_ray_kw" in code:
        pattern = r'_ray_kw = OmegaConf\.to_container.*?ray\.init\(\*\*_ray_kw\)'
        new_code, n = re.subn(pattern, CANONICAL, code, count=1, flags=re.DOTALL)
        if n > 0:
            fpath.write_text(new_code)
            print("PATCHED: main_ppo.py — replaced existing _ray_kw block with canonical")
        else:
            print("WARNING: _ray_kw found but regex did not match — manual check needed")
    else:
        print("WARNING: could not find ray.init call — manual check needed")
PATCH_RAY

# Pin training to TRAIN_GPU (judge is on JUDGE_GPU)
export CUDA_VISIBLE_DEVICES=$TRAIN_GPU
echo "Training pinned to GPU $TRAIN_GPU (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"

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
    data.val_max_samples=16 \
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.85 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.load_format=safetensors \
    ++actor_rollout_ref.rollout.attention_backend=flashinfer \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.prompt_length=2048 \
    actor_rollout_ref.rollout.response_length=6144 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    \
    critic.enable=false \
    \
    reward_model.enable=False \
    \
    custom_reward_function.path="$PROJECT_ROOT/scripts/self_play/agentic_reward.py" \
    custom_reward_function.name=async_compute_score \
    \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    trainer.total_epochs=$EPOCHS \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=medserl-agentic-small \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    ++ray_kwargs.ray_init.include_dashboard=False \
    ++ray_kwargs.ray_init.num_cpus=8 \
    "++ray_kwargs.ray_init._temp_dir=$RAY_TMPDIR_PATH" \
    ++ray_kwargs.ray_init.object_store_memory=500000000 \
    2>&1 | tee "$TRAIN_LOG"

TRAIN_EXIT=${PIPESTATUS[0]}

# ─── Step 3: Verification ─────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "=== Step 3: Verification ==="
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

# Kill judge server if we launched it
if [ -n "$VLLM_PID" ]; then
    echo ""
    echo "Stopping judge server (PID $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
fi

exit $TRAIN_EXIT
