#!/usr/bin/env bash
# Run the v6 SFT-scaling experiment: 5 cumulative shards × 2 variants + base-model eval.
#
# Variant 1 — MIXED (assessor + injector):
#   Source: data_processed/medrect_v2/mixed_sft_train.jsonl  (4,424 rows)
#   Heldout pool added at 100%: mixed_sft_heldout_rl.jsonl   (1,006 rows)
#   Counts: 1086, 2172, 3258, 4424, 5430
#     - shard 04 (4,424) is content-equivalent to mixed_sft_train.jsonl
#       → reproduces the F1 0.540 SFT v2 baseline (key sanity check).
#     - shard 05 (5,430) is the full mixed_sft_all.jsonl pool.
#
# Variant 2 — ASSESSOR-ONLY:
#   Source: data_processed/medrect_v2/assessor_sft_all.jsonl (2,585 rows)
#   Counts: 517, 1034, 1551, 2107, 2585
#     - parallel fractions (20/40/60/81.5/100 %) so curves are comparable.
#
# 0 % point: base Qwen3-4B eval, run once (--eval-base-model).
#
# Phase 2 (vLLM eval) is automatically called after Phase 1 for each variant.
# After both variants finish, run aggregate_sft_scaling_v6.py for the F1 table.

set -euo pipefail

cd "$(dirname "$0")/../.."

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/local_training/sft_scaling_v6}"
RESULTS_ROOT="${RESULTS_ROOT:-results/sft_scaling_v6}"
SEED="${SEED:-42}"
COMMON_ARGS=(
    --model-name        "$MODEL_NAME"
    --output-root       "$OUTPUT_ROOT"
    --results-root      "$RESULTS_ROOT"
    --seed              "$SEED"
    --eval-base-model
    --skip-existing
    --dataset           all
)

# ── Variant 1: mixed (assessor + injector) ───────────────────────────────────
echo
echo "=================================================================="
echo " v6 SFT scaling — variant 1: MIXED (assessor + injector)"
echo "=================================================================="
python3 scripts/medrect/run_sft_scaling_experiment.py \
    --train-file    data_processed/medrect_v2/mixed_sft_train.jsonl \
    --heldout-file  data_processed/medrect_v2/mixed_sft_heldout_rl.jsonl \
    --custom-counts "1086,2172,3258,4424,5430" \
    --experiment-name v6_mixed \
    "${COMMON_ARGS[@]}"

# ── Variant 2: assessor-only ─────────────────────────────────────────────────
echo
echo "=================================================================="
echo " v6 SFT scaling — variant 2: ASSESSOR-ONLY"
echo "=================================================================="
python3 scripts/medrect/run_sft_scaling_experiment.py \
    --train-file    data_processed/medrect_v2/assessor_sft_all.jsonl \
    --custom-counts "517,1034,1551,2107,2585" \
    --experiment-name v6_assessor_only \
    "${COMMON_ARGS[@]}"

# ── Aggregate ────────────────────────────────────────────────────────────────
echo
echo "=================================================================="
echo " v6 SFT scaling — aggregating both variants"
echo "=================================================================="
python3 scripts/medrect/aggregate_sft_scaling_v6.py \
    --results-root "$RESULTS_ROOT"
