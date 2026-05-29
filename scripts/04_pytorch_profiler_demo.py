"""
Script 04: PyTorch Profiler Demo
----------------------------------
CONCEPT: nvidia-smi tells you IF there is a problem. The PyTorch Profiler
tells you WHERE the time is going — which operators, which layers, how much
was CPU vs GPU.

This script trains a small CNN for a few steps, profiles it, and prints:
  - Top operators by CUDA (GPU) time
  - Top operators by CPU time
  - A note about what to look for

Run:
    python 04_pytorch_profiler_demo.py

Works on CPU (produces a CPU-only profile) or GPU (produces full CUDA profile).
A TensorBoard trace is saved to ./prof_output/ for optional GUI viewing.
"""

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader, TensorDataset
import os

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE   = 32
PROFILE_STEPS = 10   # only profile this many steps — we want a small window

print("=" * 60)
print("PYTORCH PROFILER DEMO")
print("=" * 60)
print(f"Device       : {DEVICE}")
print(f"Profile steps: {PROFILE_STEPS}")
print()

# ── A small CNN (like a tiny ResNet block) ────────────────────────────────────
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.AdaptiveAvgPool2d((4, 4))
        self.fc    = nn.Linear(64 * 4 * 4, 10)
        self.relu  = nn.ReLU()

    def forward(self, x):
        with record_function("conv_block_1"):
            x = self.relu(self.conv1(x))
        with record_function("conv_block_2"):
            x = self.relu(self.conv2(x))
        with record_function("pool_and_fc"):
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x

model     = TinyCNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Synthetic data
images = torch.randn(200, 3, 32, 32)
labels = torch.randint(0, 10, (200,))
loader = DataLoader(TensorDataset(images, labels), batch_size=BATCH_SIZE)

# ── Profiling setup ───────────────────────────────────────────────────────────
activities = [ProfilerActivity.CPU]
if DEVICE == "cuda":
    activities.append(ProfilerActivity.CUDA)

os.makedirs("prof_output", exist_ok=True)

print("Profiling training loop ...")
print()

with profile(
    activities=activities,
    record_shapes=True,
    profile_memory=True,
    with_stack=False,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./prof_output"),
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=PROFILE_STEPS)
) as prof:
    for step, (x, y) in enumerate(loader):
        if step >= PROFILE_STEPS + 2:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        with record_function("forward"):
            out  = model(x)
            loss = criterion(out, y)
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer_step"):
            optimizer.step()
        prof.step()

# ── Print results ─────────────────────────────────────────────────────────────
print("=" * 60)
print("TOP OPERATORS BY CPU TIME")
print("=" * 60)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

if DEVICE == "cuda":
    print("=" * 60)
    print("TOP OPERATORS BY CUDA (GPU) TIME")
    print("=" * 60)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print()
print("HOW TO READ THIS TABLE:")
print("  - 'Name'         : the operator or function (your record_function labels show up here)")
print("  - 'CPU total'    : wall time spent on CPU for this op across all steps")
print("  - 'CUDA total'   : time the GPU spent executing this op's kernels")
print("  - 'Self CPU'     : time in this op excluding its children")
print()
print("WHAT TO LOOK FOR:")
print("  - If CPU time >> CUDA time → work is not properly offloaded to GPU")
print("  - Large 'aten::copy_' entries → frequent host-to-device copies")
print("  - 'DataLoader' showing high CPU time → data pipeline bottleneck")
print()
print(f"TensorBoard trace saved to: ./prof_output/")
print("  To view: tensorboard --logdir ./prof_output")
