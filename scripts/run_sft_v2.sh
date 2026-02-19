#!/bin/bash
# Run full SFT v2 pipeline in the background:
#   1. Patch benign chains (mark all is_valid=True)
#   2. Combine all chain files into all_chains_v2.jsonl
#   3. Train 4B and/or 8B model
#
# Usage:
#   bash scripts/run_sft_v2.sh          # train both 4b and 8b
#   bash scripts/run_sft_v2.sh 4b       # train 4b only
#   bash scripts/run_sft_v2.sh 8b       # train 8b only

set -e

# Ensure screen is installed
if ! command -v screen &> /dev/null; then
    echo "screen not found — installing via apt-get..."
    apt-get update -qq && apt-get install -y screen
    echo "screen installed."
fi

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$REPO_ROOT"

MODEL_SIZE="${1:-both}"
LOG_DIR="$REPO_ROOT/outputs/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/sft_v2_${TIMESTAMP}.log"

BENIGN_FILE="data_processed/medprm_chains/medprm_chains_benign_20260218_223837.jsonl"
ASSESSOR_FILE="data_processed/medprm_chains/medprm_chains_20260128_152521.jsonl"
INJECTOR_PARTIAL="data_processed/medprm_chains/medprm_chains_20260218_162948.jsonl"
INJECTOR_FULL="data_processed/medprm_chains/medprm_chains_20260218_181331.jsonl"
COMBINED_FILE="data_processed/medprm_chains/all_chains_v2.jsonl"

run_pipeline() {
    echo "=============================================="
    echo " MedSeRL SFT v2 Pipeline"
    echo " Model: ${MODEL_SIZE}"
    echo " Started: $(date)"
    echo "=============================================="

    # Step 1: Patch benign chains
    echo ""
    echo "[1/4] Patching benign chains (mark all is_valid=True)..."
    python3 -c "
import json
chains = [json.loads(l) for l in open('${BENIGN_FILE}')]
for c in chains:
    c['is_valid'] = True
    c['validation_errors'] = None
with open('${BENIGN_FILE}', 'w') as f:
    for c in chains:
        f.write(json.dumps(c) + '\n')
print(f'  Patched {len(chains)} benign chains')
"

    # Step 2: Combine all chain files
    echo ""
    echo "[2/4] Combining chain files -> ${COMBINED_FILE}..."
    cat "${ASSESSOR_FILE}" \
        "${INJECTOR_PARTIAL}" \
        "${INJECTOR_FULL}" \
        "${BENIGN_FILE}" \
        > "${COMBINED_FILE}"

    TOTAL=$(wc -l < "${COMBINED_FILE}")
    echo "  Combined: ${TOTAL} chains total"

    # Step 3: Train 4B
    if [[ "${MODEL_SIZE}" == "4b" || "${MODEL_SIZE}" == "both" ]]; then
        echo ""
        echo "[3/4] Training Qwen3-4B LoRA..."
        python scripts/sft/train_medprm_lora.py \
            --model-size 4b \
            --train-file "${COMBINED_FILE}" \
            --output-dir outputs/local_training/qwen3-4b-medprm-v2 \
            --num-train-epochs 3
        echo "  4B training complete: $(date)"
    fi

    # Step 4: Train 8B
    if [[ "${MODEL_SIZE}" == "8b" || "${MODEL_SIZE}" == "both" ]]; then
        echo ""
        echo "[4/4] Training Qwen3-8B LoRA..."
        python scripts/sft/train_medprm_lora.py \
            --model-size 8b \
            --train-file "${COMBINED_FILE}" \
            --output-dir outputs/local_training/qwen3-8b-medprm-v2 \
            --num-train-epochs 3
        echo "  8B training complete: $(date)"
    fi

    echo ""
    echo "=============================================="
    echo " ALL DONE: $(date)"
    echo "=============================================="
}

SESSION="sft_v2"

# Abort if session already exists
if screen -list | grep -q "${SESSION}"; then
    echo "Screen session '${SESSION}' already exists!"
    echo "Attach:  screen -r ${SESSION}"
    echo "Kill:    screen -S ${SESSION} -X quit"
    exit 1
fi

# Write pipeline to a temp script so screen can run it cleanly
PIPELINE_SCRIPT=$(mktemp /tmp/sft_v2_XXXXXX.sh)
cat > "$PIPELINE_SCRIPT" << HEREDOC
#!/bin/bash
set -e
cd ${REPO_ROOT}

BENIGN_FILE="${BENIGN_FILE}"
ASSESSOR_FILE="${ASSESSOR_FILE}"
INJECTOR_PARTIAL="${INJECTOR_PARTIAL}"
INJECTOR_FULL="${INJECTOR_FULL}"
COMBINED_FILE="${COMBINED_FILE}"
MODEL_SIZE="${MODEL_SIZE}"

$(declare -f run_pipeline)
run_pipeline 2>&1 | tee "${LOG_FILE}"
HEREDOC

echo "Launching SFT v2 in screen session '${SESSION}'..."
echo "Log: ${LOG_FILE}"
echo ""
screen -dmS "${SESSION}" bash "$PIPELINE_SCRIPT"

echo "Screen session started."
echo ""
echo "Attach:   screen -r ${SESSION}"
echo "Detach:   Ctrl-A then D"
echo "Monitor:  tail -f ${LOG_FILE}"
echo "Kill:     screen -S ${SESSION} -X quit"
