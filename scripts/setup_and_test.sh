#!/bin/bash
# MedSeRL Setup and Quick Test Script
# Creates a virtual environment and runs a quick test with 16 samples

set -e

echo "=========================================="
echo "MedSeRL Setup and Quick Test"
echo "=========================================="

# Create virtual environment
if [ ! -d "med_serl" ]; then
    echo "Creating virtual environment 'med_serl'..."
    python3 -m venv med_serl
else
    echo "Virtual environment 'med_serl' already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source med_serl/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch transformers accelerate datasets pandas numpy hypothesis pytest tqdm

# Optional: Install full requirements (may take longer)
# pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Running Quick Test (16 samples)"
echo "=========================================="

# Run the quick test
python3 scripts/quick_test.py --batch_size 16 --num_episodes 2

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment later:"
echo "  source med_serl/bin/activate"
echo ""
echo "To run a quick test:"
echo "  python3 scripts/quick_test.py --batch_size 16"
echo ""
echo "To run all unit tests:"
echo "  python3 -m pytest tests/ -v"
