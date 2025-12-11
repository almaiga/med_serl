#!/bin/bash
# Launcher script to run MedGemma training in a screen session

set -e

# Install screen if not already installed
if ! command -v screen &> /dev/null; then
    echo "Installing screen..."
    apt-get update && apt-get install -y screen
fi

# Screen session name
SESSION_NAME="medgemma_training"

# Check if session already exists
if screen -list | grep -q "${SESSION_NAME}"; then
    echo "Screen session '${SESSION_NAME}' already exists!"
    echo "To attach: screen -r ${SESSION_NAME}"
    echo "To kill existing session: screen -S ${SESSION_NAME} -X quit"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Start Ray in the background (outside screen)
echo "Starting Ray cluster..."
ray start --head --node-ip-address 0.0.0.0 --dashboard-port 8265

# Wait a moment for Ray to fully start
sleep 3

# Create screen session and run training
echo "Starting training in screen session '${SESSION_NAME}'..."
screen -dmS "${SESSION_NAME}" bash -c "
    cd ${SCRIPT_DIR}
    bash train_medgemma_4b_medec_seed500.sh 2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log
"

echo ""
echo "=========================================="
echo "Training launched in screen session: ${SESSION_NAME}"
echo "=========================================="
echo ""
echo "Useful commands:"
echo "  Attach to session:  screen -r ${SESSION_NAME}"
echo "  Detach from session: Ctrl+A then D"
echo "  List sessions:      screen -ls"
echo "  Kill session:       screen -S ${SESSION_NAME} -X quit"
echo ""
echo "Monitor Ray dashboard: http://localhost:8265"
echo ""
