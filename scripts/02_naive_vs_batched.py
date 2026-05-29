"""
Script 02: Naive (one-by-one) vs Batched GPU Processing
---------------------------------------------------------
CONCEPT: GPUs are designed for massive parallelism. Sending work one item at a
time wastes the hardware — like hiring a 10,000-person factory and giving them
one widget to build at a time.

This script measures the same total work done two ways:
  - Naive:     50,000 individual forward passes (one sample each)
  - Batched:   One forward pass over all 50,000 samples at once

Run on a GPU node:
    python 02_naive_vs_batched.py

Run on CPU (for testing, numbers will differ but the ratio stays clear):
    python 02_naive_vs_batched.py
"""

import time
import torch

FEATURE_SIZE = 1024    # dimension of each input vector
DATA_SIZE    = 10_000  # total number of samples to process
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 55)
print("NAIVE vs BATCHED GPU PROCESSING")
print("=" * 55)
print(f"Device      : {DEVICE}")
print(f"Samples     : {DATA_SIZE:,}")
print(f"Feature dim : {FEATURE_SIZE}")
print()

# Build a simple model and move it to the device
model = torch.nn.Linear(FEATURE_SIZE, FEATURE_SIZE).to(DEVICE)
model.eval()

# ── Approach 1: Naive — one sample at a time ──────────────────────────────────
print("[1] Naive approach: processing one sample at a time ...")
torch.cuda.synchronize() if DEVICE == "cuda" else None

start = time.perf_counter()
with torch.no_grad():
    for _ in range(DATA_SIZE):
        sample = torch.randn(1, FEATURE_SIZE, device=DEVICE)
        _ = model(sample)

torch.cuda.synchronize() if DEVICE == "cuda" else None
naive_time = time.perf_counter() - start
print(f"    Total time : {naive_time:.3f} s")
print(f"    Per sample : {naive_time / DATA_SIZE * 1000:.4f} ms")

# ── Approach 2: Batched — all samples at once ─────────────────────────────────
print()
print("[2] Batched approach: all samples in one call ...")

# Pre-allocate the full batch on the device
all_samples = torch.randn(DATA_SIZE, FEATURE_SIZE, device=DEVICE)

torch.cuda.synchronize() if DEVICE == "cuda" else None

start = time.perf_counter()
with torch.no_grad():
    _ = model(all_samples)

torch.cuda.synchronize() if DEVICE == "cuda" else None
batched_time = time.perf_counter() - start
print(f"    Total time : {batched_time:.3f} s")
print(f"    Per sample : {batched_time / DATA_SIZE * 1000:.4f} ms")

# ── Summary ──────────────────────────────────────────────────────────────────
speedup = naive_time / batched_time
print()
print("=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"  Naive time   : {naive_time:.3f} s")
print(f"  Batched time : {batched_time:.3f} s")
print(f"  Speedup      : {speedup:.1f}×")
print()
print("WHY: Each GPU call has a fixed overhead (kernel launch, PCIe transfer).")
print("     Batching amortizes that overhead over many samples at once.")
print()
print("TAKEAWAY: Never loop over samples and send them to the GPU one at a time.")
print("          Always batch your data.")
