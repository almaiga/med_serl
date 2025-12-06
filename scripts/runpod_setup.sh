#!/bin/bash
# MedSeRL - Fully Automated RunPod Setup Script
# No manual intervention required

set -e  # Exit on any error

echo "=========================================="
echo "MedSeRL - Automated Environment Setup"
echo "=========================================="
echo ""

# Configuration - Everything in /workspace for persistence across pod restarts
ENV_NAME="med_serl"
INSTALL_DIR="/workspace/miniconda3"
ENV_DIR="${INSTALL_DIR}/envs/${ENV_NAME}"
HF_HOME_DIR="/workspace/.huggingface"
HF_TOKEN_PATH="/workspace/.hf_token"
PIP_CACHE_DIR="/workspace/.pip_cache"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

# Change to project directory
cd "${PROJECT_DIR}"
echo "Working directory: $(pwd)"
echo ""

# Ensure /workspace exists
if [ ! -d "/workspace" ]; then
    echo "❌ /workspace not found. Are you running on RunPod?"
    exit 1
fi

# Step 0: Install screen for background processes
echo "Step 0: Installing screen..."
if command -v screen &> /dev/null; then
    echo "✓ screen already installed"
else
    echo "Installing screen via apt-get..."
    apt-get update -qq && apt-get install -y -qq screen
    echo "✓ screen installed successfully"
fi
echo ""

# Step 1: Install Miniconda3 if not present
echo "Step 1: Checking for Miniconda3..."
if [ -f "${INSTALL_DIR}/bin/conda" ]; then
    echo "✓ Conda already installed at: ${INSTALL_DIR}"
else
    echo "Installing Miniconda3..."
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    
    wget -q "${MINICONDA_URL}" -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "${INSTALL_DIR}"
    rm /tmp/miniconda.sh
    
    # Initialize conda for current shell (no restart needed)
    eval "$(${INSTALL_DIR}/bin/conda shell.bash hook)"
    
    # Add to bashrc for future sessions
    ${INSTALL_DIR}/bin/conda init bash 2>/dev/null || true
    
    echo "✓ Miniconda3 installed successfully"
fi

# Always initialize conda for current session
eval "$(${INSTALL_DIR}/bin/conda shell.bash hook)"
echo ""

# Step 2: Create conda environment (fully automated, no prompts)
echo "Step 2: Setting up conda environment '${ENV_NAME}'..."
if [ -d "${ENV_DIR}" ]; then
    echo "✓ Environment already exists at ${ENV_DIR}"
    echo "  (To recreate, manually run: rm -rf ${ENV_DIR})"
else
    echo "Creating new environment..."
    conda create -y -p "${ENV_DIR}" python=3.10
    echo "✓ Environment created at ${ENV_DIR}"
fi
echo ""

# Step 3: Activate environment and install requirements
echo "Step 3: Installing Python packages..."
conda activate "${ENV_DIR}"

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Use pip cache in /workspace for persistence
mkdir -p "${PIP_CACHE_DIR}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"

# Upgrade pip
pip install --upgrade pip -q

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip install --cache-dir "${PIP_CACHE_DIR}" -r requirements.txt
    echo "✓ Requirements installed"
else
    echo "⚠️  requirements.txt not found, installing core packages..."
    pip install --cache-dir "${PIP_CACHE_DIR}" torch transformers accelerate datasets pandas numpy hypothesis pytest tqdm wandb
    echo "✓ Core packages installed"
fi
echo ""

# Step 4: Hugging Face authentication (saved in /workspace for persistence)
echo "Step 4: Configuring Hugging Face authentication..."
mkdir -p "${HF_HOME_DIR}"
export HF_HOME="${HF_HOME_DIR}"

if [ -n "${HF_TOKEN}" ]; then
    # Token provided via environment variable - save it for future restarts
    echo "${HF_TOKEN}" > "${HF_TOKEN_PATH}"
    chmod 600 "${HF_TOKEN_PATH}"
    echo "✓ HF token saved from environment variable to ${HF_TOKEN_PATH}"
elif [ -f "${HF_TOKEN_PATH}" ]; then
    # Token already saved from previous session
    export HF_TOKEN=$(cat "${HF_TOKEN_PATH}")
    echo "✓ HF token loaded from ${HF_TOKEN_PATH} (persisted from previous session)"
else
    # Prompt for token and save it
    echo "Hugging Face token not found."
    echo "(Get your token from: https://huggingface.co/settings/tokens)"
    echo ""
    read -p "Enter HF Token (or press Enter to skip): " HF_TOKEN_INPUT
    
    if [ -n "${HF_TOKEN_INPUT}" ]; then
        echo "${HF_TOKEN_INPUT}" > "${HF_TOKEN_PATH}"
        chmod 600 "${HF_TOKEN_PATH}"
        export HF_TOKEN="${HF_TOKEN_INPUT}"
        echo "✓ HF token saved to ${HF_TOKEN_PATH} (will persist across pod restarts)"
    else
        echo "⚠️  Skipped. You can add it later: echo 'your_token' > ${HF_TOKEN_PATH}"
    fi
fi
echo ""

# Step 5: Verify installation
echo "Step 5: Verifying installation..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import transformers; print(f'✓ Transformers {transformers.__version__}')"
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    python -c "import torch; print(f'✓ GPU: {torch.cuda.get_device_name(0)}')"
fi
echo ""

echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Everything is installed in /workspace (persists across pod restarts):"
echo "  - Miniconda: ${INSTALL_DIR}"
echo "  - Environment: ${ENV_DIR}"
echo "  - HF cache: ${HF_HOME_DIR}"
echo "  - Pip cache: ${PIP_CACHE_DIR}"
echo "  - HF token: ${HF_TOKEN_PATH}"
echo ""
echo "To activate the environment in a new shell:"
echo "  source ${INSTALL_DIR}/bin/activate ${ENV_DIR}"
echo ""
echo "Quick commands:"
echo "  python scripts/quick_test.py --batch_size 16"
echo "  python -m pytest tests/ -v"
echo ""
