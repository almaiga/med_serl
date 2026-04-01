#!/bin/bash

set -euo pipefail

FAST_ROOT="${FAST_ROOT:-/home/dpsk_a2a/DeepEP}"
VERL_DIR="${VERL_DIR:-$FAST_ROOT/verl}"
MED_SERL_DIR="${MED_SERL_DIR:-/workspace/med_serl}"
VERL_REPO="${VERL_REPO:-https://github.com/verl-project/verl.git}"

echo "=================================================="
echo "MedSeRL SGLang Pod Bootstrap"
echo "=================================================="
echo "Fast root:   $FAST_ROOT"
echo "VERL dir:    $VERL_DIR"
echo "MedSeRL dir: $MED_SERL_DIR"
echo "=================================================="

if [ ! -d "$MED_SERL_DIR" ]; then
    echo "ERROR: med_serl repo not found at $MED_SERL_DIR"
    exit 1
fi

mkdir -p "$FAST_ROOT"

if [ ! -d "$VERL_DIR/.git" ]; then
    echo "Cloning VERL into fast local storage..."
    git clone "$VERL_REPO" "$VERL_DIR"
else
    echo "Refreshing existing VERL checkout..."
    git -C "$VERL_DIR" fetch --all --prune
    git -C "$VERL_DIR" checkout main
    git -C "$VERL_DIR" pull --ff-only
fi

echo ""
echo "Installing editable VERL without touching image-pinned dependencies..."
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip uninstall -y verl >/dev/null 2>&1 || true
(
    cd "$VERL_DIR"
    python3 -m pip install --no-deps -e .
)

echo ""
echo "Installing MedSeRL runtime extras that are not rollout-backend specific..."
python3 -m pip install --upgrade --upgrade-strategy only-if-needed \
    aiofiles \
    aiohttp \
    datasets \
    jsonlines \
    openai \
    pandas \
    pyarrow \
    python-dotenv \
    pynvml \
    tensorboard \
    wandb \
    peft \
    loralib \
    bitsandbytes \
    optimum \
    "math-verify[antlr4_13_2]" \
    rouge_score

echo ""
echo "Version check..."
python3 - <<'PY'
import importlib

def version(name):
    try:
        mod = importlib.import_module(name)
        print(f"{name}={getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name}=IMPORT_FAILED ({exc})")

for pkg in ("torch", "ray", "transformers", "sglang", "verl", "pyarrow"):
    version(pkg)
PY

echo ""
echo "Bootstrap complete."
echo "Recommended first launch on a clean SGLang image:"
echo "  cd $MED_SERL_DIR"
echo "  SKIP_DATAGEN=1 bash scripts/self_play/run_multiturn_training.sh"
echo ""
echo "If you need to regenerate chained parquet from scratch, note that"
echo "scripts/self_play/generate_chained_data.py still uses offline vLLM."
