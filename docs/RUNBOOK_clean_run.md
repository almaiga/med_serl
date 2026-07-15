# Clean self-play run — runbook

Every expensive action has a verify step. Follow top to bottom; if a gate fails, stop.

## What was fixed vs v6
- **Judge thinking ON** (`judge_client.py` detection branch + `medrect_judge_prompts.json` max_tokens=2048). v6 trained against a thinking-OFF judge that was blind to plausible errors (0/6 medium, 0/6 subtle). Fixed judge: 92% (44/48), verified thinking active (47/48, avg 509 reasoning tokens).
- **Verified HF pushes** (`hf_push_verified.py` + `checkpoint_watcher.sh`). v6 checkpoints/adapters were lost because pushes silently failed. Now every push is re-listed and confirmed.
- **RESUME_MODE=auto, KEEP_ONLY_LATEST_CHECKPOINT=0** baked into the launcher.

## Architecture
- **Judge**: MedRECT-32B (~61 GB) on its own pod/GPU, reachable via `JUDGE_VLLM_URL`. Thinking-on ≈ 5 s/call — this is the wall-clock bottleneck; a fast GPU here helps most.
- **Policy**: Qwen3-4B on 2× training GPUs (A100 80 GB is best value; needs ~57 GB/GPU).

---

## Step 0 — sync + offline preflight (free)
```bash
cd /workspace/med_serl && git pull
bash scripts/self_play/preflight_clean_run.sh          # Tiers 1-2, must be all [OK]
```

## Step 1 — back up data to HF (free, verified)
```bash
bash scripts/self_play/backup_data_to_hf.sh            # medrect_v2 + synthetic_test, private, verified
```

## Step 2 — start the judge server (on the judge pod/GPU)
Start MedRECT-32B as a vLLM OpenAI server, then export its URL for the training pod:
```bash
export JUDGE_VLLM_URL="http://<judge-host>:8002/v1/chat/completions"
```

## Step 3 — judge thinking gate on GPU (~2 min, cheap)
```bash
RUN_JUDGE_TEST=1 bash scripts/self_play/preflight_clean_run.sh   # must print "judge thinking end-to-end: GO"
```

## Step 4 — smoke self-play (cheap gate; tests judge on REAL injector output)
```bash
JUDGE_VLLM_URL="$JUDGE_VLLM_URL" bash scripts/self_play/smoke_selfplay.sh
```
Read the game-log health at the end:
- `judge_status=ok` HIGH (>90 %)
- `SAME-on-error` LOW (<15 %)
- `game_invalid` LOW (<15 %)

If all three good → proceed. If not → STOP (cheap to find here).

## Step 5 — start the verified checkpoint pusher (separate terminal, BEFORE launch)
```bash
OUTPUT_DIR=outputs/self_play_v7 REPO_PREFIX=Abdine/qwen3-4b-medserl-v7-step \
  PRIVATE=1 nohup bash scripts/self_play/checkpoint_watcher.sh \
  > logs/checkpoint_watcher.log 2>&1 &
```
Each checkpoint is pushed to `Abdine/qwen3-4b-medserl-v7-step<N>` and verified. Tail `logs/checkpoint_watcher.log` and confirm "step N VERIFIED on HF".

## Step 6 — launch the full run
```bash
JUDGE_VLLM_URL="$JUDGE_VLLM_URL" bash scripts/self_play/launch_clean_selfplay.sh
```

## Step 7 — monitor
```bash
watch -n 30 bash scripts/self_play/monitor_training.sh
```
Watch for: reward trend up, `judge_status=ok` high, `SAME-on-error` low, no grad_norm spikes. WARN flags print automatically.

## Step 8 — post-run eval (one clean harness, both thinking modes)
```bash
for step in $(ls outputs/self_play_v7 | grep -oE '[0-9]+' | sort -n); do
  for mode in thinking no-thinking; do
    python3 scripts/medrect/inference_detection_vllm.py \
      --model_path outputs/self_play_v7/global_step_${step}/actor/huggingface \
      --dataset all --mode $mode \
      --output_dir results/v7_eval/step${step}__${mode}
  done
done
```
Pick the best checkpoint by MEDEC accuracy (the fair metric), confirm it's already on HF (checkpoint_watcher pushed it), done.

## If the pod dies
Data is on HF (Step 1). Checkpoints are on HF (Step 5). Restart the pod, `git pull`, `hf download` the latest checkpoint, resume with `RESUME_MODE=auto`.
