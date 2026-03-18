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

N_EXAMPLES="${N_EXAMPLES:-30}"
ROLE="${ROLE:-assessor}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# ── Find most recent interactions file ───────────────────────────────────────
if [ -z "$INTERACTIONS" ]; then
    INTERACTIONS=$(find "$PROJECT_ROOT/outputs/self_play" -name "interactions_*.jsonl" \
        2>/dev/null | sort | tail -1)
fi

if [ -z "$INTERACTIONS" ]; then
    echo "ERROR: No interactions_*.jsonl found under outputs/self_play/"
    echo "Run a smoke test first: bash scripts/self_play/run_reinforce_smoke.sh"
    exit 1
fi

echo "=================================================="
echo "  MedSeRL Judge Test"
echo "  Interactions : $INTERACTIONS"
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
        echo "  Set it to your judge pod's URL: export JUDGE_VLLM_URL=http://<ip>:8002/v1/chat/completions"
    fi
fi
echo ""

python3 -m scripts.self_play.test_agentic_judge \
    --from-interactions "$INTERACTIONS" \
    --n "$N_EXAMPLES" \
    --role "$ROLE" \
    $DRY_RUN_FLAG

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "  Exit code: $EXIT_CODE"
echo "  Results saved to: results/self_play/judge_test_results.jsonl"
echo "  Judge traces:     results/self_play/judge_traces/"
echo "=================================================="

exit $EXIT_CODE
