#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Explorer — Single V100 GPU Training Job (template)
#
#  Copy this file, adjust the values marked with ← CHANGE, and submit with:
#      sbatch v100_single_gpu.slurm
#
#  Before submitting, run this job in an interactive session first:
#      srun --partition=gpu-short --gres=gpu:v100:1 --cpus-per-task=8 --mem=32G \
#           --time=01:00:00 --pty bash
#  Confirm GPU utilisation > 70% with:  watch -n 1 nvidia-smi
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=my_training           # ← CHANGE: descriptive name
#SBATCH --partition=gpu                   # gpu (8 h) for production training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                 # ← CHANGE: set to num_workers + 1
#SBATCH --gres=gpu:v100:1                 # request 1 × V100 specifically
#SBATCH --mem=32G                         # CPU RAM (NOT GPU VRAM)
#SBATCH --time=04:00:00                   # ← CHANGE: expected runtime + 20%
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Create the log directory before SLURM writes to it
mkdir -p logs

# ── Environment ───────────────────────────────────────────────────────────────
# No CUDA module needed — PyTorch's CUDA runtime ships in the wheels installed
# by setup_env.sh.  Just load Python and activate the venv.
module purge
module load python/3.11
source /scratch/$USER/Optimizing-GPU-Usage-on-Explorer/gpu_training_env/bin/activate  # ← CHANGE

echo "========================================"
echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : $SLURMD_NODENAME"
echo "GPUs          : $CUDA_VISIBLE_DEVICES"
echo "Start time    : $(date)"
echo "========================================"

# ── Verify GPU allocation (always) ───────────────────────────────────────────
python scripts/01_gpu_verify.py

# ── Copy dataset to node-local storage if available ──────────────────────────
# $TMPDIR (node-local NVMe) is much faster than Lustre for small-file reads.
# Fall back to /scratch if $TMPDIR is not set on this node.
if [ -n "$TMPDIR" ] && [ -d "$TMPDIR" ]; then
    echo "Copying dataset to node-local storage ($TMPDIR) ..."
    rsync -a /scratch/$USER/mydata/ $TMPDIR/mydata/   # ← CHANGE: your data path
    DATA_DIR=$TMPDIR/mydata
    echo "Done. Using DATA_DIR=$DATA_DIR"
else
    DATA_DIR=/scratch/$USER/mydata                    # ← CHANGE: your data path
    echo "TMPDIR not available, using DATA_DIR=$DATA_DIR"
fi

# ── Training ──────────────────────────────────────────────────────────────────
python train.py \                                     # ← CHANGE: your script
    --data-dir $DATA_DIR \
    --epochs 50 \
    --batch-size 256 \
    --fp16                                # FP16 on V100 (use --bf16 on A100/H100)

echo "========================================"
echo "End time : $(date)"
echo "========================================"

# ── Review efficiency after the job finishes ─────────────────────────────────
# Run this after the job completes:
#   gpu-logs $SLURM_JOB_ID
# Look for average GPU utilisation > 70%.  If lower, see Section 7 of the training.
