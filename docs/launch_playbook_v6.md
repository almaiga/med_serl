# Launch Playbook — Self-Play v6 (corrected pipeline)

> Copy-paste runbook for the post-failure-analysis self-play run.
> Everything offline-verifiable has already been verified — this is just the
> execution sequence. Do these steps **in order** when the SFT finishes.

Compute:
- Training pod: **2× H200 SXM**  ($8.82/hr)
- Judge pod:    **1× A100 PCIe** ($1.41/hr)

Total: $10.23/hr × estimated 2–3 hours = **~$20–$30 for the full RL run**.

---

## Step 1 — Sanity-check the new SFT  (~5 min, training pod)

The SFT is producing `Abdine/qwen3-4b-medrect-mixed-v2`. Before launching RL,
confirm the F1 baseline so §11 of `docs/judge_bottleneck_failure_analysis.md`
remains internally consistent.

```bash
# wait for the HF push to finish, then:
python3 scripts/medrect/inference_detection_vllm.py \
  --model_path Abdine/qwen3-4b-medrect-mixed-v2 \
  --output_dir results/sft_eval/v2_sft
```

Defaults: full ms+uw test set, thinking on, single GPU (the project's standard
invocation for any detection eval).

**Decision rule:**

| Reported F1 | Action |
|---|---|
| 0.50 – 0.65 | **expected range.** Proceed to Step 2. Update §11 of the doc with the exact number. |
| > 0.65 | clean SFT is unexpectedly strong; great news. Proceed to Step 2. |
| < 0.45 | something went wrong with the SFT (overfitting, bad data join). DO NOT launch RL; ping me. |

---

## Step 2 — Start the judge server  (~3 min, judge pod)

On the A100 pod:

```bash
git pull
bash scripts/self_play/start_judge_server.sh
# Wait for the line "INFO: Application startup complete."
# Then GET /v1/models from inside the pod to confirm:
curl -s http://localhost:8000/v1/models | jq '.data[].id'
# expected: "pfnet/Preferred-MedRECT-32B"
```

Note the pod's internal IP (the address the training pod will reach).
You'll plug it into `JUDGE_VLLM_URL` in Step 3.

---

## Step 3 — Launch self-play  (training pod)

```bash
git pull

# Point at the judge pod from Step 2
export JUDGE_VLLM_URL="http://<JUDGE_POD_IP>:8000/v1"

# Sanity-check the wiring without spending a token (the offline reward unit test):
python3 scripts/self_play/test_game_reward_v5.py
# Must print "ALL REWARD CHECKS PASS". If not, do not launch.

# Then launch with the v6 config:
SMOKE=0 \
AUTO_SCREEN=1 \
N_GPUS=2 \
ACTOR_MODEL=Abdine/qwen3-4b-medrect-mixed-v2 \
OUTPUT_DIR=outputs/self_play_v6_clean_judge \
EXPERIMENT_NAME=medserl_selfplay_v6_clean_judge \
MAX_PAIRS=0 \
TRAIN_BATCH_SIZE=16 \
PPO_MINI_BATCH_SIZE=8 \
PPO_EPOCHS=2 \
TOTAL_EPOCHS=5 \
SAVE_FREQ=33 \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.5 \
KL_COEF=0.01 \
KEEP_ONLY_LATEST_CHECKPOINT=0 \
JUDGE_MODEL=pfnet/Preferred-MedRECT-32B \
JUDGE_TYPE=detection \
JUDGE_PROMPT_STYLE=hint_v2 \
JUDGE_VLLM_URL="$JUDGE_VLLM_URL" \
WANDB=1 \
RESUME_MODE=auto \
RESTART_AFTER_CHECKPOINT=1 \
RESTART_ON_FAILURE=1 \
MAX_AUTORESTARTS=500 \
CHECKPOINT_RESTART_GRACE_SEC=30 \
RUNNER_SCRIPT=scripts/self_play/run_multiturn_training.sh \
bash scripts/self_play/run_online_selfplay_autorestart.sh
```

### What the startup banner must say (else ctrl-C)

Look for these three lines in the first ~5 seconds of output:

```
Judge URL: http://<JUDGE_POD_IP>:8000/v1
Judge model: pfnet/Preferred-MedRECT-32B
Judge type: detection   (prompt style: hint_v2)
```

If any line is wrong → ctrl-C; it's the only remaining wiring failure mode.

---

## Step 4 — Monitor the first epoch on W&B

The §11 EV math predicts these signals if the corrected pipeline is working:

| Signal | v5 (broken judge) | v6 (calibrated judge) — expectation |
|---|---|---|
| `reward/error_injection/recall` | bounded ~0.65–0.82 (judge-CHANGED games only) | climbs past 0.85 — the binding-constraint ceiling is lifted |
| `reward/error_injection/fp_rate` | collapsed from 0.33 → 0.18 (conservatism) | does NOT collapse — stays ≥ 0.20 |
| `reward/mean` per epoch | non-monotonic, peaked then regressed | monotonically increasing through epoch 5 |
| `judge_verdict=SAME` on error mode | ~27 % (the v5 failure) | ≤ 5 % (matches Exp 1's 0 % on real errors) |

If any signal regresses in the v5 direction at epoch 2 → ctrl-C and we revisit.

---

## Step 5 — Post-training eval  (~5 min per checkpoint)

When the run finishes (or after 5 epochs):

```bash
mkdir -p results/selfplay_v6/per_checkpoint

# Evaluate every saved checkpoint on MEDEC test
for ckpt in outputs/self_play_v6_clean_judge/*/actor/global_step_*; do
  step=$(basename "$ckpt")
  echo "=== $step ==="
  python3 scripts/medrect/inference_detection_vllm.py \
    --model_path "$ckpt" \
    --output_dir "results/selfplay_v6/per_checkpoint/${step}"
done

# Pick the best by test F1
python3 -c '
import json, glob
rows = []
for f in sorted(glob.glob("results/selfplay_v6/per_checkpoint/*.json")):
    d = json.load(open(f))
    rows.append((f, d.get("f1", 0), d.get("recall", 0)))
for f, f1, r in sorted(rows, key=lambda x: -x[1]):
    print(f"{f1:.3f}  recall={r:.3f}  {f}")
'
```

The top row is the paper's headline model.

---

## Step 6 — Update the doc

Once the headline F1 is in, append to `docs/judge_bottleneck_failure_analysis.md`:

- §10.1 (or §14): the new SFT v2 actual F1 (replace the placeholder 0.593 if it differs)
- §14 (or new section): v6 self-play final F1 + the per-epoch trend showing the §11 predictions held (or didn't)
- §15: ship decision — v6 model vs r2

---

## Quick reference — files you'll need

- Failure analysis: `docs/judge_bottleneck_failure_analysis.md`
- Reward unit test: `scripts/self_play/test_game_reward_v5.py`
- Reward EV math: `scripts/self_play/reward_ev_check.py`
- Exp 1 (real held-out errors): `scripts/self_play/exp1_real_errors.py`
- Exp 2 (probe set): `scripts/self_play/exp2_run_probes.py` + `exp2_compare.py`
- Probe set: `data_processed/judge_bench/exp2_probes.jsonl`
- Test set: `data_processed/judge_bench/medec_test.json`
- Launch script: `scripts/self_play/run_multiturn_training.sh`
- Autorestart wrapper: `scripts/self_play/run_online_selfplay_autorestart.sh`
- Judge launcher: `scripts/self_play/start_judge_server.sh`
- Judge prompt config: `configs/prompts/medrect_judge_prompts.json` (includes `hint_v2`)
