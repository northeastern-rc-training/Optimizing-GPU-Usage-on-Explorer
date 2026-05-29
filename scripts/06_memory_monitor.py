"""
Script 06: GPU Memory Monitoring During Training
-------------------------------------------------
CONCEPT: nvidia-smi shows memory at one moment in time. During training,
memory usage evolves — it grows as the model loads, then again when the
optimizer state allocates, then peaks during the backward pass.

This script prints a memory report at each epoch so you can watch the
growth pattern and identify when you're close to OOM.

It also demonstrates the "two axes" problem:
  - Utilization % and Memory % are INDEPENDENT metrics
  - High memory + low utilization = something is wrong
  - High memory + high utilization = healthy and working hard

Run:
    python 06_memory_monitor.py    (GPU recommended; runs on CPU with no memory stats)
"""

import torch
import torch.nn as nn
import time

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS     = 4
BATCH_SIZE = 64
STEPS_PER_EPOCH = 20

print("=" * 65)
print("GPU MEMORY MONITOR DEMO")
print("=" * 65)
print(f"Device : {DEVICE}")
print()

if DEVICE == "cuda":
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Total VRAM : {total_vram_gb:.1f} GB")
    print()


def report_memory(tag: str):
    """Print current and peak VRAM usage with a label."""
    if DEVICE != "cuda":
        print(f"  [{tag}] (no CUDA — memory stats unavailable on CPU)")
        return
    allocated_mb = torch.cuda.memory_allocated() / 1024**2
    reserved_mb  = torch.cuda.memory_reserved() / 1024**2
    peak_mb      = torch.cuda.max_memory_allocated() / 1024**2
    total_mb     = torch.cuda.get_device_properties(0).total_memory / 1024**2
    pct          = allocated_mb / total_mb * 100
    bar_len      = 30
    bar_fill     = int(bar_len * pct / 100)
    bar          = "█" * bar_fill + "░" * (bar_len - bar_fill)
    print(f"  [{tag}]")
    print(f"    Allocated : {allocated_mb:6.0f} MB  ({pct:4.1f}%)  [{bar}]")
    print(f"    Reserved  : {reserved_mb:6.0f} MB  (PyTorch cache — not all in use)")
    print(f"    Peak ever : {peak_mb:6.0f} MB")
    print()


# ── Model + optimizer ─────────────────────────────────────────────────────────
class TrainableNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))


print("--- Before model load ---")
report_memory("before model")

model = TrainableNet().to(DEVICE)
report_memory("after model.to(device)")

optimizer = torch.optim.Adam(model.parameters())
report_memory("after optimizer init")

criterion = nn.CrossEntropyLoss()

# ── Training loop with memory reports ────────────────────────────────────────
print("=" * 65)
print("TRAINING LOOP")
print("=" * 65)

for epoch in range(1, EPOCHS + 1):
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for step in range(STEPS_PER_EPOCH):
        x = torch.randn(BATCH_SIZE, 3, 32, 32, device=DEVICE)
        y = torch.randint(0, 10, (BATCH_SIZE,), device=DEVICE)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    elapsed = time.perf_counter() - t0
    print(f"Epoch {epoch}/{EPOCHS}  ({elapsed:.2f}s)")
    report_memory(f"epoch {epoch} end")

# ── Final analysis ────────────────────────────────────────────────────────────
if DEVICE == "cuda":
    peak_mb  = torch.cuda.max_memory_allocated() / 1024**2
    total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print("=" * 65)
    print("MEMORY ANALYSIS")
    print("=" * 65)
    print(f"  Peak VRAM used : {peak_mb:.0f} MB / {total_mb:.0f} MB  ({peak_mb/total_mb*100:.1f}%)")
    print()
    if peak_mb / total_mb < 0.5:
        print("  ▶ Memory usage is LOW. Consider increasing batch size.")
        print("    Larger batches → better GPU utilization → faster training.")
    elif peak_mb / total_mb < 0.85:
        print("  ▶ Memory usage looks HEALTHY (50–85% of VRAM).")
        print("    You have a little headroom — try a slightly larger batch.")
    else:
        print("  ▶ Memory usage is HIGH (>85%). You are close to OOM.")
        print("    Options: reduce batch size, enable mixed precision,")
        print("             or use gradient checkpointing.")
    print()
    print("USEFUL SNIPPET: add to your training script")
    print()
    print("  torch.cuda.reset_peak_memory_stats()")
    print("  # ... your training epoch ...")
    print("  peak  = torch.cuda.max_memory_allocated() / 1024**3")
    print("  total = torch.cuda.get_device_properties(0).total_memory / 1024**3")
    print("  print(f'Peak VRAM: {peak:.1f} GB / {total:.1f} GB  ({100*peak/total:.0f}%)')")
