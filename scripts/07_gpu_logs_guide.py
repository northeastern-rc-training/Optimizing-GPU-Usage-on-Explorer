"""
Script 07 — gpu-logs Interpretation Guide
-------------------------------------------
`gpu-logs` is Explorer's post-job tool for reviewing how efficiently a job used
its GPU.  After a batch job finishes it reports the time-series of GPU
utilisation (and memory) over the whole run — the post-job equivalent of
watching `nvidia-smi` live during the job.

How to use it after your job finishes:
    gpu-logs <jobid>

This script does NOT call gpu-logs directly (it is a cluster command that only
works on Explorer for your own completed jobs).  Instead it prints an
ILLUSTRATIVE gpu-logs-style time-series and explains how to read it and what
corrective actions to take when the numbers are low.

Usage:
    python 07_gpu_logs_guide.py

Note: the numbers below are illustrative.  Your real `gpu-logs <jobid>` output
is the source of truth — this guide only teaches you how to interpret it.
"""

SEP    = "=" * 66
SUBSEP = "-" * 66

# Illustrative job summary
SIMULATED_JOB = {
    "Job ID"             : "2374457",
    "Partition"          : "gpu",
    "GPUs requested"     : 1,
    "CPUs per task"      : 8,
    "Requested walltime" : "08:00:00",
    "Actual walltime"    : "06:12:33",
    "GPU"                : "Tesla V100-SXM2-32GB",
    "GPU memory total"   : 32.0,   # GB
}

# Illustrative time-series: (minutes into job, GPU util %, VRAM used GB).
# This pattern — utilisation bouncing between low values — is the classic
# data-starvation signature: the GPU works in short bursts, then waits.
SERIES = [
    (0,   5, 1.0),
    (30, 22, 3.0),
    (60, 18, 3.1),
    (90, 41, 3.1),
    (120, 16, 3.2),
    (150, 24, 3.1),
    (180, 19, 3.2),
    (210, 45, 3.1),
    (240, 21, 3.2),
    (270, 17, 3.1),
    (300, 23, 3.2),
    (330, 20, 3.1),
    (360, 26, 3.2),
]

SPARK = "▁▂▃▄▅▆▇█"


def spark(values, lo=0, hi=100):
    """Render a list of numbers as a unicode sparkline."""
    out = []
    for v in values:
        frac = (v - lo) / (hi - lo) if hi > lo else 0
        idx = min(len(SPARK) - 1, max(0, int(frac * (len(SPARK) - 1))))
        out.append(SPARK[idx])
    return "".join(out)


def bar(pct, width=30, warn_below=70):
    filled = int(width * pct / 100)
    symbol = "█" if pct >= warn_below else "▒"
    return symbol * filled + "░" * (width - filled)


def fmt_pct(pct, warn_below=70, good_above=80):
    if pct >= good_above:
        return f"{pct:.1f}%  ✓ GOOD"
    elif pct >= warn_below:
        return f"{pct:.1f}%  ⚑ OK, but room to improve"
    else:
        return f"{pct:.1f}%  ✗ LOW"


j = SIMULATED_JOB
utils = [u for _, u, _ in SERIES]
vrams = [v for _, _, v in SERIES]
avg_util = sum(utils) / len(utils)
peak_util = max(utils)
avg_vram = sum(vrams) / len(vrams)
vram_pct = avg_vram / j["GPU memory total"] * 100

print(SEP)
print("  gpu-logs — POST-JOB GPU UTILISATION REPORT (ILLUSTRATIVE)")
print("  Real usage:  gpu-logs <jobid>  after your sbatch job finishes")
print(SEP)
print()
print(f"  Job ID        : {j['Job ID']}")
print(f"  Partition     : {j['Partition']}")
print(f"  GPU           : {j['GPU']}")
print(f"  GPUs          : {j['GPUs requested']}")
print(f"  CPUs/task     : {j['CPUs per task']}")
print(f"  Requested     : {j['Requested walltime']}")
print(f"  Actual        : {j['Actual walltime']}")
print()

# ── Time-series ───────────────────────────────────────────────────────────────
print(SUBSEP)
print("  GPU UTILISATION OVER TIME")
print(SUBSEP)
print(f"    {spark(utils)}   (each cell ≈ 30 min)")
print(f"    min {min(utils)}%   mean {avg_util:.0f}%   peak {peak_util}%")
print()
print("    Per-sample:")
for minutes, u, v in SERIES:
    ubar = "█" * int(u / 4) + "░" * (25 - int(u / 4))
    print(f"      t+{minutes:>3} min  [{ubar}] {u:>3}%   VRAM {v:.1f} GB")
print()

# ── Headline metrics ──────────────────────────────────────────────────────────
print(SUBSEP)
print("  SUMMARY METRICS")
print(SUBSEP)
print(f"  GPU utilisation (avg) :")
print(f"    {fmt_pct(avg_util)}")
print(f"    [{bar(avg_util)}]")
print()
print(f"  GPU memory (avg)      :")
print(f"    {avg_vram:.1f} GB / {j['GPU memory total']:.1f} GB  = {fmt_pct(vram_pct)}")
print(f"    [{bar(vram_pct)}]")
print()

# ── Interpretation guide ──────────────────────────────────────────────────────
print(SEP)
print("  HOW TO INTERPRET THE REPORT")
print(SEP)
print("""
  GPU UTILISATION (most important metric)
  ────────────────────────────────────────
  What it measures:
    The fraction of time GPU kernels were actually running, over the whole
    job.  A flat high line is healthy; a line that bounces between low values
    (like the one above) means the GPU keeps running out of work and waiting.

  Target: > 70% average

  If low (< 50%):
    • Check your data pipeline first (Section 4 of the training).
      num_workers=0 is the single most common cause.
    • Is your data on /home?  Move it to /scratch or copy to $TMPDIR at
      job start.
    • Are you requesting multiple GPUs when one would do?  Each idle GPU
      drags down your effective efficiency.

  GPU MEMORY
  ───────────
  What it measures:
    Average VRAM used.  A low percentage means unused capacity you could
    spend on a larger batch.

  Target: 75–90%

  If low (< 40%) — as in the report above (~9%):
    • Increase batch size.  More VRAM usage almost always raises utilisation.
    • Enable mixed precision (BF16 on A100/H100, FP16 on V100): it frees
      memory so you can push batch size higher, which often improves
      utilisation even as memory % drops.

  READING THE SHAPE OF THE CURVE
  ───────────────────────────────
    • Sawtooth / bouncing low  → data starvation (fix the DataLoader).
    • Flat and low             → work too small for the GPU (scale batch/model).
    • Flat and high (> 80%)    → healthy; you are compute-bound.
    • High with periodic dips  → sync stalls (multi-GPU) or per-epoch overhead.
""")

# ── Action plan ────────────────────────────────────────────────────────────────
print(SEP)
print("  ACTION PLAN FOR THIS ILLUSTRATIVE JOB")
print(SEP)
print(f"  GPU util avg = {avg_util:.0f}% (target: > 70%), VRAM ~{vram_pct:.0f}%")
print()
print("  1. IMMEDIATE: The bouncing-low curve is textbook data starvation.")
print("     Confirm the DataLoader uses num_workers > 0 and pin_memory=True.")
print("     Script 03 benchmarks your configuration.")
print()
print(f"  2. VRAM is only ~{vram_pct:.0f}% used — increase batch size substantially.")
print("     Script 06 shows the util gain from a right-sized workload.")
print()
print("  3. Move the dataset off /home to /scratch, and copy to $TMPDIR at")
print("     job start for the fastest reads.")
print()
print("  RULE: Run `gpu-logs <jobid>` after EVERY batch job.  If average")
print("  utilisation stays below 60%, fix it before submitting the next run —")
print("  efficient jobs keep queue times low for everyone on Explorer.")
print(SEP)
