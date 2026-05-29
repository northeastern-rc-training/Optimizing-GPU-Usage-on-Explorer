#!/bin/bash
# Setup script for the GPU Profiling training environment.
# Run this ONCE to create the virtualenv and install dependencies.
#
# Usage:
#   chmod +x setup_profiling_env.sh
#   ./setup_profiling_env.sh
#
# After setup, activate with:
#   source profiling_env/bin/activate

set -e

VENV_DIR="profiling_env"

echo "=== GPU Profiling Training Environment Setup ==="
echo ""

# Create venv if it does not exist
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/3] Creating virtual environment: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
else
    echo "[1/3] Virtual environment already exists: $VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"
echo "[2/3] Activated: $(which python)"

# Install dependencies
echo "[3/3] Installing dependencies ..."
pip install --quiet --upgrade pip

# On Explorer (GPU node), load the CUDA-enabled torch:
#   pip install torch torchvision
# For CPU-only testing (local machine / login node):
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch torchvision --quiet

echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To test GPU availability:"
echo "  python scripts/01_gpu_verify.py"
