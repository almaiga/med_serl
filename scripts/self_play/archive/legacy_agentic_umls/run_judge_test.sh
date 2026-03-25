#!/bin/bash
# Run the Agentic UMLS Judge test against real training outputs.
#
# Run this on the TRAINING pod (Instance A).
# The judge vLLM server must be running on a separate pod (Instance B).
#
# Usage:
#   # Dry-run (no judge server needed):
#   bash scripts/self_play/run_judge_test.sh --dry-run
#
#   # Full test (judge server required):
#   JUDGE_VLLM_URL=http://<instance-B-ip>:8002/v1/chat/completions \
#   bash scripts/self_play/run_judge_test.sh
#
# Env overrides:
#   JUDGE_VLLM_URL   — URL of the judge vLLM server
#   UMLS_API_KEY     — UMLS API key (required for UMLS lookups)
#   N_EXAMPLES       — Number of examples to test (default: 30)
#   INTERACTIONS     — Explicit path to interactions JSONL (auto-detected if unset)
#   ROLE             — Which role to test: assessor or injector (default: assessor)
#   PARQUET          — Path to verl Parquet for note context (auto-detected if unset)

N_EXAMPLES="${N_EXAMPLES:-30}"
ROLE="${ROLE:-assessor}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Auto-detect parquet for judge context enrichment
if [ -z "$PARQUET" ]; then
    PARQUET=$(find "$PROJECT_ROOT/data_processed/self_play" \
        -name "train_grpo.parquet" -o -name "train.parquet" 2>/dev/null | head -1)
fi

# ── Find most recent interactions file ───────────────────────────────────────
if [ -z "$INTERACTIONS" ]; then
    # Check both reward_function.py's default log dir and outputs/
    INTERACTIONS=$(find \
        "$PROJECT_ROOT/results/self_play/interactions" \
        "$PROJECT_ROOT/outputs/self_play" \
        -name "interactions_*.jsonl" 2>/dev/null \
        -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
fi

if [ -z "$INTERACTIONS" ]; then
    echo "ERROR: No interactions_*.jsonl found."
    echo "Searched: results/self_play/interactions/ and outputs/self_play/"
    echo "Run a smoke test first: bash scripts/self_play/run_reinforce_smoke.sh"
    exit 1
fi

echo "=================================================="
echo "  MedSeRL Judge Test"
echo "  Interactions : $INTERACTIONS"
echo "  Parquet      : ${PARQUET:-(none — no note context enrichment)}"
echo "  N examples   : $N_EXAMPLES"
echo "  Role filter  : $ROLE"
echo "  Judge URL    : ${JUDGE_VLLM_URL:-http://localhost:8002/v1/chat/completions}"
echo "  UMLS key     : $([ -n "$UMLS_API_KEY" ] && echo 'set' || echo 'NOT SET')"
echo "=================================================="
echo ""

# Parse flags
DRY_RUN_FLAG=""
for arg in "$@"; do
    [ "$arg" = "--dry-run" ] && DRY_RUN_FLAG="--dry-run"
done

if [ -n "$DRY_RUN_FLAG" ]; then
    echo "=== DRY RUN (no judge server or UMLS needed) ==="
else
    echo "=== FULL TEST (judge server + UMLS) ==="
    if [ -z "$JUDGE_VLLM_URL" ]; then
        echo "  WARNING: JUDGE_VLLM_URL not set, defaulting to http://localhost:8002"
        echo "  Set it to your judge pod's internal DNS URL:"
        echo "    export JUDGE_VLLM_URL=http://<judge-pod-id>.runpod.internal:8002/v1/chat/completions"
        echo "  Both pods must have Global Networking enabled in RunPod dashboard."
    fi

    # Pre-flight connectivity check
    MODELS_URL="${JUDGE_VLLM_URL:-http://localhost:8002/v1/chat/completions}"
    MODELS_URL="${MODELS_URL%/chat/completions}/models"
    echo "  Checking judge server at $MODELS_URL ..."
    if curl -sf --max-time 8 "$MODELS_URL" > /dev/null 2>&1; then
        echo "  ✓ Judge server reachable"
    else
        echo "  ✗ Cannot reach judge server."
        echo "    Is Global Networking enabled on both pods?"
        echo "    Is the judge server running? (bash scripts/self_play/start_judge_server.sh)"
        exit 1
    fi
fi
echo ""

PARQUET_FLAG=""
[ -n "$PARQUET" ] && PARQUET_FLAG="--parquet $PARQUET"

python3 -m scripts.self_play.test_agentic_judge \
    --from-interactions "$INTERACTIONS" \
    --n "$N_EXAMPLES" \
    --role "$ROLE" \
    $PARQUET_FLAG \
    $DRY_RUN_FLAG

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "  Exit code: $EXIT_CODE"
echo "  Results saved to: results/self_play/judge_test_results.jsonl"
echo "  Judge traces:     results/self_play/judge_traces/"
echo "=================================================="

exit $EXIT_CODE
