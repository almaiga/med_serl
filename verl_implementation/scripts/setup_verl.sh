#!/bin/bash
# Setup script for verl environment
#
# This script installs verl and its dependencies.
#
# Usage:
#   bash setup_verl.sh [--backend vllm|sglang]

set -e

BACKEND="${BACKEND:-vllm}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "Setting up verl with $BACKEND backend"
echo "============================================"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

# Clone verl if not present
VERL_DIR="${VERL_DIR:-$HOME/verl}"

if [ ! -d "$VERL_DIR" ]; then
    echo "Cloning verl repository..."
    git clone https://github.com/volcengine/verl.git "$VERL_DIR"
else
    echo "verl directory exists at $VERL_DIR"
    echo "Pulling latest changes..."
    cd "$VERL_DIR" && git pull
fi

cd "$VERL_DIR"

# Install verl with chosen backend
echo "Installing verl with $BACKEND backend..."
if [ "$BACKEND" = "vllm" ]; then
    pip install -e ".[vllm]"
elif [ "$BACKEND" = "sglang" ]; then
    pip install -e ".[sglang]"
else
    echo "Unknown backend: $BACKEND"
    exit 1
fi

# Install additional dependencies
echo "Installing additional dependencies..."
pip install pandas pyarrow wandb transformers accelerate

# Verify installation
echo "============================================"
echo "Verifying installation..."
python3 -c "import verl; print(f'verl version: {verl.__version__}')" || echo "verl installed (no version attribute)"
python3 -c "import vllm; print(f'vllm version: {vllm.__version__}')" || echo "vllm not installed (expected if using sglang)"
python3 -c "import torch; print(f'torch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

echo "============================================"
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Preprocess MEDEC data:"
echo "   python data/preprocess_medec.py --input_dir /path/to/MEDEC --output_dir ~/data/medec"
echo ""
echo "2. Run training:"
echo "   bash scripts/run_medserl_reinforce.sh --model Qwen/Qwen2.5-3B-Instruct --gpus 1"
echo "============================================"
