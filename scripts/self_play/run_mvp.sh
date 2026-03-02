#!/bin/bash
# =============================================================================
# MedSeRL — Agentic UMLS Judge MVP
#
# Launches:
#   1. A vLLM judge server on port 8001 (Qwen3-4B for entity extraction + adjudication)
#   2. The test harness against 50 self-play examples
#
# Prerequisites:
#   - UMLS_API_KEY set in environment (or in .env at project root)
#   - vllm installed:  pip install vllm
#   - aiohttp installed: pip install aiohttp
#   - data at data_processed/self_play/train.parquet (run preprocess_medec.py first)
#
# Usage:
#   # Full pipeline test (50 examples, requires UMLS_API_KEY + GPU):
#   bash scripts/self_play/run_mvp.sh
#
#   # Dry run (no GPU, no UMLS — tests plumbing only):
#   bash scripts/self_play/run_mvp.sh --dry-run
#
#   # Custom example count:
#   bash scripts/self_play/run_mvp.sh --n 10
#
#   # Extraction-only test (just Step 1):
#   bash scripts/self_play/run_mvp.sh --extraction-only
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Defaults (override via env vars)
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-4B}"
JUDGE_PORT="${JUDGE_PORT:-8001}"
JUDGE_GPU="${JUDGE_GPU:-0}"
N_EXAMPLES="${N_EXAMPLES:-50}"
DRY_RUN=false
EXTRACTION_ONLY=false

# Parse args
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)    DRY_RUN=true; shift ;;
        --extraction-only) EXTRACTION_ONLY=true; shift ;;
        --n)          N_EXAMPLES="$2"; shift 2 ;;
        --model)      JUDGE_MODEL="$2"; shift 2 ;;
        --port)       JUDGE_PORT="$2"; shift 2 ;;
        --gpu)        JUDGE_GPU="$2"; shift 2 ;;
        *)            EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

export JUDGE_VLLM_URL="http://localhost:${JUDGE_PORT}/v1/chat/completions"
export JUDGE_MODEL="$JUDGE_MODEL"

echo "============================================="
echo "  MedSeRL — Agentic UMLS Judge MVP"
echo "============================================="
echo "  Judge Model:  $JUDGE_MODEL"
echo "  Judge URL:    $JUDGE_VLLM_URL"
echo "  N Examples:   $N_EXAMPLES"
echo "  Dry Run:      $DRY_RUN"
echo "  UMLS API Key: ${UMLS_API_KEY:+set}"
echo "============================================="

# ─── Dry run: skip vLLM server ─────────────────────────────────────────────
if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "Running dry test (no vLLM server, no UMLS calls)..."
    python -m scripts.self_play.test_agentic_judge \
        --dry-run \
        --n "$N_EXAMPLES" \
        $EXTRA_ARGS
    exit 0
fi

# ─── Check UMLS API key ────────────────────────────────────────────────────
if [ -z "${UMLS_API_KEY:-}" ]; then
    # Try loading from .env
    if [ -f "$PROJECT_ROOT/.env" ]; then
        export $(grep -E '^UMLS_API_KEY=' "$PROJECT_ROOT/.env" | xargs)
    fi
    if [ -z "${UMLS_API_KEY:-}" ]; then
        echo "WARNING: UMLS_API_KEY not set. UMLS lookups will return empty evidence."
        echo "         Set it via: export UMLS_API_KEY=your-key-here"
        echo "         Or add UMLS_API_KEY=your-key to .env at project root."
        echo ""
    fi
fi

# ─── Check data exists ─────────────────────────────────────────────────────
DATA_PATH="data_processed/self_play/train.parquet"
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: $DATA_PATH not found."
    echo "Run preprocessing first:"
    echo "  python -m scripts.self_play.preprocess_medec"
    exit 1
fi

# ─── Launch vLLM judge server ──────────────────────────────────────────────
JUDGE_PID=""

cleanup() {
    if [ -n "$JUDGE_PID" ] && kill -0 "$JUDGE_PID" 2>/dev/null; then
        echo ""
        echo "Stopping judge server (PID $JUDGE_PID)..."
        kill "$JUDGE_PID" 2>/dev/null || true
        wait "$JUDGE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Check if a server is already running on the port
if curl -s "http://localhost:${JUDGE_PORT}/v1/models" > /dev/null 2>&1; then
    echo "Judge server already running on port $JUDGE_PORT — reusing."
else
    echo ""
    echo "Starting vLLM judge server on port $JUDGE_PORT (GPU $JUDGE_GPU)..."
    CUDA_VISIBLE_DEVICES="$JUDGE_GPU" python -m vllm.entrypoints.openai.api_server \
        --model "$JUDGE_MODEL" \
        --port "$JUDGE_PORT" \
        --max-model-len 4096 \
        --dtype auto \
        --trust-remote-code \
        --disable-log-requests \
        > "$PROJECT_ROOT/outputs/logs/judge_server.log" 2>&1 &
    JUDGE_PID=$!
    echo "Judge server PID: $JUDGE_PID"
    echo "Waiting for server to be ready..."

    # Wait up to 120s for the server
    for i in $(seq 1 60); do
        if curl -s "http://localhost:${JUDGE_PORT}/v1/models" > /dev/null 2>&1; then
            echo "Judge server ready! (waited ${i}x2s)"
            break
        fi
        if ! kill -0 "$JUDGE_PID" 2>/dev/null; then
            echo "ERROR: Judge server died. Check outputs/logs/judge_server.log"
            exit 1
        fi
        sleep 2
    done

    if ! curl -s "http://localhost:${JUDGE_PORT}/v1/models" > /dev/null 2>&1; then
        echo "ERROR: Judge server failed to start after 120s."
        echo "Check outputs/logs/judge_server.log for errors."
        exit 1
    fi
fi

# ─── Run the test ──────────────────────────────────────────────────────────
echo ""
echo "Running Agentic UMLS Judge test ($N_EXAMPLES examples)..."

if [ "$EXTRACTION_ONLY" = true ]; then
    python -m scripts.self_play.test_agentic_judge \
        --extraction-only \
        --n "$N_EXAMPLES" \
        $EXTRA_ARGS
else
    python -m scripts.self_play.test_agentic_judge \
        --n "$N_EXAMPLES" \
        $EXTRA_ARGS
fi

echo ""
echo "Done! Check results in:"
echo "  - results/self_play/judge_test_results.jsonl"
echo "  - results/self_play/judge_traces/"
