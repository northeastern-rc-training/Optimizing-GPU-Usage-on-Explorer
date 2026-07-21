#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  GPU Profiling Training — Environment Setup (Explorer)
#  GPU Profiling and Performance Optimization on Explorer
#
#  Run this script ONCE to create the Python virtual environment used
#  throughout the training demos.
#
#  Steps:
#    1. Get an interactive GPU node:
#         srun --partition=gpu-short --gres=gpu:1 --cpus-per-task=4 --mem=16G \
#              --time=01:00:00 --pty bash
#
#    2. Load the required modules:
#         module load cuda/12.2 python/3.11
#
#    3. Run this script from your scratch directory:
#         cd /scratch/$USER
#         git clone https://github.com/northeastern-rc-training/Optimizing-GPU-Usage-on-Explorer.git
#         cd Optimizing-GPU-Usage-on-Explorer
#         chmod +x setup_profiling_env.sh
#         ./setup_profiling_env.sh
#
#    4. Activate the environment:
#         source profiling_env/bin/activate
#
#  Tip: add the module loads to your ~/.bashrc so they load automatically in
#  future sessions (optional):
#    module load cuda/12.2 python/3.11
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

VENV_DIR="profiling_env"

echo "======================================================="
echo "  GPU Profiling Training — Environment Setup (Explorer)"
echo "======================================================="

# ── Prerequisites ─────────────────────────────────────────────────────────────
echo ""
echo "Checking prerequisites ..."

if ! command -v python3 &>/dev/null; then
    echo ""
    echo "ERROR: python3 not found."
    echo "Load the module first:  module load python/3.11"
    exit 1
fi

if ! command -v nvcc &>/dev/null; then
    echo ""
    echo "WARNING: nvcc not found.  PyTorch will still install, but if you are"
    echo "on a GPU node CUDA may not be visible.  Load the module:"
    echo "  module load cuda/12.2"
fi

echo "python3 : $(python3 --version)"
echo "nvcc    : $(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' || echo 'not loaded')"
echo ""

# ── Create venv (skip if it already exists) ───────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] Creating virtual environment: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
else
    echo "[1/4] Virtual environment already exists: $VENV_DIR"
    echo "      (Delete it with 'rm -rf $VENV_DIR' to rebuild from scratch.)"
fi

# ── Activate ──────────────────────────────────────────────────────────────────
source "$VENV_DIR/bin/activate"
echo "[2/4] Activated: $(which python)"

pip install --quiet --upgrade pip

# ── Install PyTorch ───────────────────────────────────────────────────────────
# On Explorer (GPU node), the default CUDA-enabled wheels work:
#   pip install torch torchvision torchaudio
# For CPU-only testing (local machine / login node), use:
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
echo "[3/4] Installing PyTorch (CUDA-enabled wheels) ..."
pip install --quiet torch torchvision torchaudio

# ── Install profiling / monitoring tools ──────────────────────────────────────
# nvitop        : live GPU dashboard (Section 2)
# tensorboard   : view PyTorch Profiler traces (Section 3)
# nvidia-ml-py  : NVML bindings for live utilisation sampling (Section 5, script 06)
# matplotlib    : optional plotting
echo "[4/4] Installing profiling and monitoring tools ..."
pip install --quiet nvitop tensorboard nvidia-ml-py matplotlib numpy

echo ""
echo "======================================================="
echo "  Environment setup complete."
echo "======================================================="
echo ""
echo "Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Quick verification (run on a GPU node):"
echo "  python -c \"import torch; print(torch.cuda.is_available())\""
echo "  python scripts/01_gpu_verify.py"
echo ""
echo "IMPORTANT — run from /scratch, not /home:"
echo "  cd /scratch/\$USER/Optimizing-GPU-Usage-on-Explorer"
echo "  source $VENV_DIR/bin/activate"
echo "  python scripts/01_gpu_verify.py"
echo "======================================================="
