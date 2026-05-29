"""
Script 01: Verify GPU Allocation
---------------------------------
Run this at the TOP of every GPU job to confirm you have what you requested.
Works on both GPU nodes and CPU-only machines (will report "no CUDA").

On a GPU node:
    python 01_gpu_verify.py

Expected output on a GPU node:
    CUDA available: True
    GPU count: 1
      GPU 0: NVIDIA A100-SXM4-80GB  80 GB
    CUDA_VISIBLE_DEVICES: 0
    ...nvidia-smi output...
"""

import os
import subprocess
import sys

# ── Check 1: Is CUDA (the GPU driver bridge) reachable? ───────────────────────
try:
    import torch
    cuda_available = torch.cuda.is_available()
except ImportError:
    print("PyTorch is not installed in this environment.")
    sys.exit(1)

print("=" * 55)
print("GPU VERIFICATION REPORT")
print("=" * 55)

print(f"\n[1] CUDA available : {cuda_available}")
print(f"    PyTorch version : {torch.__version__}")

if not cuda_available:
    print("\n  CUDA is NOT available. Most likely causes:")
    print("  (a) You are on a login node — request a compute node first:")
    print("      srun --partition=gpu --gres=gpu:1 --pty bash")
    print("  (b) The cuda module is not loaded:")
    print("      module load cuda/12.2")
    print("  (c) Your SLURM --gres directive has a typo.")
    print("\n  Nothing below this line will be meaningful without a GPU.")
    print("=" * 55)
    sys.exit(0)

# ── Check 2: How many GPUs, and which ones? ────────────────────────────────────
gpu_count = torch.cuda.device_count()
print(f"\n[2] GPU count      : {gpu_count}")

for i in range(gpu_count):
    props = torch.cuda.get_device_properties(i)
    vram_gb = props.total_memory / 1024**3
    print(f"    GPU {i}: {props.name}")
    print(f"           VRAM : {vram_gb:.1f} GB")
    print(f"           SMs  : {props.multi_processor_count}")

# ── Check 3: CUDA_VISIBLE_DEVICES — what SLURM gave us ─────────────────────────
cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET")
print(f"\n[3] CUDA_VISIBLE_DEVICES : {cvd}")
if cvd == "NOT SET":
    print("    (This is fine for single-GPU jobs without SLURM)")

# ── Check 4: Cross-verify with nvidia-smi ─────────────────────────────────────
print("\n[4] nvidia-smi cross-check:")
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,uuid",
         "--format=csv,noheader"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        for line in result.stdout.strip().splitlines():
            print(f"    {line}")
    else:
        print("    nvidia-smi not available or returned an error.")
except FileNotFoundError:
    print("    nvidia-smi binary not found (expected on GPU nodes).")

# ── Check 5: Quick tensor round-trip test ────────────────────────────────────
print("\n[5] Quick tensor round-trip test:")
try:
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    y = (x * 2).cpu()
    print(f"    CPU → GPU → CPU: {y.tolist()}  ✓")
except Exception as e:
    print(f"    FAILED: {e}")

print("\n" + "=" * 55)
print("All checks complete. If checks 1–5 pass, your GPU is ready.")
print("=" * 55)
