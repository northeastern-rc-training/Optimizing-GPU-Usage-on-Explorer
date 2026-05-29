"""
Script 03: DataLoader Configuration Benchmark
-----------------------------------------------
CONCEPT: The GPU processes a batch in milliseconds. If loading the NEXT batch
takes longer than that, the GPU sits idle — waiting for the CPU to deliver data.
This is like a high-speed assembly line that has to stop every few seconds
because the supply truck hasn't arrived yet.

This script times a DataLoader under four configurations and computes
how much of the GPU's time would be spent waiting.

Run:
    python 03_dataloader_benchmark.py

No GPU required — this measures CPU-side data loading speed.
"""

import time
import torch
from torch.utils.data import DataLoader, TensorDataset

# ── Synthetic dataset (avoids disk I/O so we isolate loader overhead) ─────────
N_SAMPLES    = 5_000
IMAGE_SHAPE  = (3, 64, 64)   # small images; real ResNet uses (3, 224, 224)
N_BATCHES    = 50             # how many batches to time
BATCH_SIZE   = 64

print("=" * 60)
print("DATALOADER CONFIGURATION BENCHMARK")
print("=" * 60)
print(f"Dataset  : {N_SAMPLES:,} synthetic images {IMAGE_SHAPE}")
print(f"Batches  : {N_BATCHES} × batch_size={BATCH_SIZE}")
print()

# Build a synthetic dataset
images = torch.randn(N_SAMPLES, *IMAGE_SHAPE)
labels = torch.randint(0, 10, (N_SAMPLES,))
dataset = TensorDataset(images, labels)

# Simulate a realistic GPU compute time per batch (e.g., 15 ms for a small model)
SIMULATED_GPU_MS = 15.0

def time_loader(loader, n_batches, label):
    """Time n_batches iterations of loader; return mean time in ms."""
    # Warm-up pass to spawn workers
    for i, _ in enumerate(loader):
        if i == 2: break

    times = []
    for i, (x, y) in enumerate(loader):
        if i == 0:
            t0 = time.perf_counter()
        if i >= n_batches:
            break
        # Simulate moving to device (non-blocking is better, shown in config 4)
        _ = x.float()
    elapsed = time.perf_counter() - t0
    mean_ms = elapsed / n_batches * 1000
    return mean_ms

configs = [
    dict(label="Config A: num_workers=0 (default)",
         kwargs=dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)),
    dict(label="Config B: num_workers=2",
         kwargs=dict(batch_size=BATCH_SIZE, num_workers=2, pin_memory=False)),
    dict(label="Config C: num_workers=4",
         kwargs=dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=False)),
    dict(label="Config D: num_workers=4 + pin_memory=True",
         kwargs=dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=True,
                     persistent_workers=True, prefetch_factor=2)),
]

results = []
for cfg in configs:
    loader = DataLoader(dataset, **cfg["kwargs"])
    mean_ms = time_loader(loader, N_BATCHES, cfg["label"])
    results.append((cfg["label"], mean_ms))
    print(f"  {cfg['label']}")
    print(f"    Mean batch load time : {mean_ms:.1f} ms")
    # Compute GPU idle fraction
    idle_pct = max(0, (mean_ms - SIMULATED_GPU_MS) / mean_ms * 100)
    if mean_ms > SIMULATED_GPU_MS:
        print(f"    GPU would be idle    : {idle_pct:.0f}% of the time  ← DATA BOUND")
    else:
        print(f"    Loader faster than GPU ({SIMULATED_GPU_MS:.0f} ms compute) ← GPU BOUND ✓")
    print()

# ── Summary table ────────────────────────────────────────────────────────────
print("=" * 60)
print("SUMMARY (simulated GPU compute = {:.0f} ms/batch)".format(SIMULATED_GPU_MS))
print("=" * 60)
print(f"  {'Configuration':<42} {'Load (ms)':>9}  {'Outcome'}")
print(f"  {'-'*42}  {'-'*9}  {'-'*20}")
for label, ms in results:
    short = label.split(":")[0]
    outcome = "GPU idle" if ms > SIMULATED_GPU_MS else "GPU bound ✓"
    print(f"  {short:<42} {ms:>9.1f}  {outcome}")

print()
print("HPC NOTE: On a cluster, your data may live on Lustre (shared filesystem).")
print("          Move it to $TMPDIR (node-local SSD) at job start for 2-4× speedup:")
print()
print("  # In your SLURM script, before python:")
print("  rsync -a /lustre/scratch/$USER/mydata/ $TMPDIR/mydata/")
print("  python train.py --data-dir $TMPDIR/mydata/")
