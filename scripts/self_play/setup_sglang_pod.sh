#!/bin/bash

set -euo pipefail

detect_fast_root() {
    if [ -d "/sgl-workspace" ]; then
        echo "/sgl-workspace"
    elif [ -d "/workspace" ]; then
        echo "/workspace"
    elif [ -d "/home/dpsk_a2a/DeepEP" ]; then
        echo "/home/dpsk_a2a/DeepEP"
    else
        echo "$HOME"
    fi
}

detect_med_serl_dir() {
    if [ -n "${MED_SERL_DIR:-}" ] && [ -d "${MED_SERL_DIR:-}" ]; then
        echo "$MED_SERL_DIR"
    elif [ -d "/sgl-workspace/med_serl" ]; then
        echo "/sgl-workspace/med_serl"
    elif [ -d "/workspace/med_serl" ]; then
        echo "/workspace/med_serl"
    elif [ -d "$PWD/.git" ] || [ -f "$PWD/scripts/self_play/run_multiturn_training.sh" ]; then
        echo "$PWD"
    else
        echo ""
    fi
}

FAST_ROOT="${FAST_ROOT:-$(detect_fast_root)}"
VERL_DIR="${VERL_DIR:-$FAST_ROOT/verl}"
MED_SERL_DIR="$(detect_med_serl_dir)"
VERL_REPO="${VERL_REPO:-https://github.com/verl-project/verl.git}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=================================================="
echo "MedSeRL SGLang Pod Bootstrap"
echo "=================================================="
echo "Fast root:   $FAST_ROOT"
echo "VERL dir:    $VERL_DIR"
echo "MedSeRL dir: $MED_SERL_DIR"
echo "Python bin:  $PYTHON_BIN"
echo "=================================================="

if [ -z "$MED_SERL_DIR" ] || [ ! -d "$MED_SERL_DIR" ]; then
    echo "ERROR: med_serl repo not found."
    echo "Set MED_SERL_DIR explicitly, for example:"
    echo "  MED_SERL_DIR=/sgl-workspace/med_serl bash scripts/self_play/setup_sglang_pod.sh"
    exit 1
fi

mkdir -p "$FAST_ROOT"

if [ -d "$VERL_DIR/.git" ]; then
    echo "Refreshing existing VERL checkout..."
    git -C "$VERL_DIR" fetch --all --prune
    git -C "$VERL_DIR" checkout main
    git -C "$VERL_DIR" pull --ff-only
elif [ -d "$VERL_DIR" ] && [ ! -e "$VERL_DIR/.git" ]; then
    echo "ERROR: $VERL_DIR exists but is not a git checkout."
    echo "Remove it or set VERL_DIR explicitly."
    exit 1
else
    echo "Cloning VERL into fast local storage..."
    git clone "$VERL_REPO" "$VERL_DIR"
fi

echo ""
echo "Installing editable VERL without touching image-pinned dependencies..."
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
"$PYTHON_BIN" -m pip uninstall -y verl >/dev/null 2>&1 || true
(
    cd "$VERL_DIR"
    "$PYTHON_BIN" -m pip install --no-deps -e .
)

echo ""
echo "Installing MedSeRL runtime extras without breaking SGLang image pins..."
"$PYTHON_BIN" -m pip install --upgrade --upgrade-strategy only-if-needed \
    aiofiles \
    aiohttp \
    "antlr4-python3-runtime==4.9.3" \
    "nvidia-cutlass-dsl>=4.4.1" \
    datasets \
    jsonlines \
    "openai==2.6.1" \
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
    rouge_score

echo ""
echo "Version check..."
"$PYTHON_BIN" - <<'PY'
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
echo "  bash scripts/self_play/run_multiturn_training.sh"
echo ""
echo "Chained datagen now supports SGLang and vLLM backends."
