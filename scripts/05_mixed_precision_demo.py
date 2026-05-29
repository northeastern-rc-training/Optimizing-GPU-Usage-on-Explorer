"""
Script 05: Mixed Precision Demo (FP32 vs BF16)
------------------------------------------------
CONCEPT: By default, PyTorch uses 32-bit floats (FP32) everywhere.
Modern GPUs (A100, H100) have dedicated Tensor Cores that operate at much
higher throughput when using 16-bit floats (FP16 or BF16).

Switching to BF16 (preferred on A100/H100):
  - Roughly HALVES activation memory  → you can fit a larger batch
  - Unlocks Tensor Core throughput     → faster matrix operations
  - Requires one extra line of code    → autocast context manager

FP16 vs BF16:
  - FP16  : smaller numeric range, needs a "loss scaler" to prevent underflow
  - BF16  : same range as FP32, no loss scaler needed — preferred on A100/H100

Run:
    python 05_mixed_precision_demo.py   (needs CUDA for the mixed-precision part)
"""

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import time

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 128
STEPS       = 30

print("=" * 60)
print("MIXED PRECISION (FP32 vs BF16) DEMO")
print("=" * 60)
print(f"Device     : {DEVICE}")
print(f"Batch size : {BATCH_SIZE}")
print(f"Steps      : {STEPS}")
print()

if DEVICE == "cpu":
    print("NOTE: Running on CPU. Mixed precision has minimal effect on CPU.")
    print("      On an A100/H100 GPU, BF16 typically gives 1.5–3× speedup.")
    print("      The memory comparison below is still valid.")
    print()

# ── A medium-sized model (large enough to show memory difference) ─────────────
class MediumNet(nn.Module):
    def __init__(self, width=1024, depth=6):
        super().__init__()
        layers = [nn.Linear(width, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers.append(nn.Linear(width, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

INPUT_DIM = 1024
model     = MediumNet(width=INPUT_DIM).to(DEVICE)
criterion = nn.CrossEntropyLoss()

# ── Measure peak memory and throughput for FP32 ───────────────────────────────
def run_training(use_amp, amp_dtype, label):
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler_device = DEVICE if DEVICE == "cuda" else "cpu"
    scaler = GradScaler(device=scaler_device) if (use_amp and amp_dtype == torch.float16) else None

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()

    for step in range(STEPS):
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=DEVICE)
        y = torch.randint(0, 10, (BATCH_SIZE,), device=DEVICE)
        opt.zero_grad()

        if use_amp:
            amp_device = DEVICE if DEVICE == "cuda" else "cpu"
            with autocast(device_type=amp_device, dtype=amp_dtype):
                out  = model(x)
                loss = criterion(out, y)
        else:
            out  = model(x)
            loss = criterion(out, y)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    elapsed  = time.perf_counter() - start
    peak_mb  = torch.cuda.max_memory_allocated() / 1024**2 if DEVICE == "cuda" else None
    samples_per_sec = BATCH_SIZE * STEPS / elapsed

    return elapsed, peak_mb, samples_per_sec


print("[1] Running FP32 training ...")
fp32_time, fp32_mem, fp32_tput = run_training(False, None, "FP32")
print(f"    Time          : {fp32_time:.2f} s")
print(f"    Throughput    : {fp32_tput:,.0f} samples/sec")
if fp32_mem:
    print(f"    Peak VRAM     : {fp32_mem:.0f} MB")
print()

print("[2] Running BF16 autocast training ...")
bf16_time, bf16_mem, bf16_tput = run_training(True, torch.bfloat16, "BF16")
print(f"    Time          : {bf16_time:.2f} s")
print(f"    Throughput    : {bf16_tput:,.0f} samples/sec")
if bf16_mem:
    print(f"    Peak VRAM     : {bf16_mem:.0f} MB")
print()

# ── Summary ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("SUMMARY")
print("=" * 60)
speedup = fp32_time / bf16_time
print(f"  Speedup (BF16 vs FP32)      : {speedup:.2f}×")
if fp32_mem and bf16_mem:
    mem_reduction = (1 - bf16_mem / fp32_mem) * 100
    print(f"  Memory reduction (BF16)     : {mem_reduction:.0f}%")
    print(f"  FP32 peak VRAM              : {fp32_mem:.0f} MB")
    print(f"  BF16 peak VRAM              : {bf16_mem:.0f} MB")

print()
print("CODE CHANGE REQUIRED: just 3 lines")
print()
print("  # FP32 (original)")
print("  out  = model(x)")
print("  loss = criterion(out, y)")
print("  loss.backward()")
print()
print("  # BF16 (with autocast) — add this wrapper:")
print("  with autocast(device_type='cuda', dtype=torch.bfloat16):")
print("      out  = model(x)")
print("      loss = criterion(out, y)")
print("  loss.backward()   # gradients stay FP32 automatically")
print()
print("USE BF16 on A100/H100.  Use FP16 on V100/T4 (with GradScaler).")
