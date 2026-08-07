#!/bin/bash
# =============================================================================
# Fine-tune Julia Copilot on a SLURM cluster.
#
#   sbatch scripts/train.slurm
#   sbatch --export=ALL,MODEL=bigcode/starcoderbase-1b,FRAC=0.35 scripts/train.slurm
#
# The #SBATCH values below are defaults; anything passed on the command line
# overrides them, e.g.
#
#   sbatch --partition=gpu-long --time=48:00:00 --gres=gpu:a100:1 scripts/train.slurm
#
# Partition names, module names and GPU types are site-specific -- check
# `sinfo -s` and `module avail` and adjust the marked lines before first use.
# =============================================================================

#SBATCH --job-name=jl-copilot-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                  # SITE-SPECIFIC: some clusters use --gpus=1
#SBATCH --partition=gpu               # SITE-SPECIFIC
#SBATCH --mem=64G
#SBATCH --time=24:00:00
##SBATCH --account=<your_account>     # uncomment if your site requires one
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=you@example.com

set -euo pipefail

# -----------------------------------------------------------------------------
# Job parameters (override with --export=ALL,NAME=value)
# -----------------------------------------------------------------------------
MODEL="${MODEL:-HuggingFaceTB/SmolLM2-135M}"
FRAC="${FRAC:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
QUANTIZED="${QUANTIZED:-1}"           # 1 -> 4-bit NF4 + LoRA, 0 -> full fine-tune
FIRST_LINE="${FIRST_LINE:-0}"
SIMPLE_INPUT="${SIMPLE_INPUT:-0}"
DATA="${DATA:-data/combined_projects.zip}"

# Project root = the directory containing this script's parent.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs

# -----------------------------------------------------------------------------
# Caches on scratch, never in $HOME
#
# HuggingFace and uv caches routinely reach tens of GB. Home directories are
# usually quota'd and often NFS-backed, which makes model loading slow as well
# as liable to fail mid-run. Point everything at node-local or parallel scratch.
# -----------------------------------------------------------------------------
SCRATCH="${SCRATCH:-${SLURM_TMPDIR:-/scratch/$USER}}"   # SITE-SPECIFIC
export HF_HOME="${HF_HOME:-$SCRATCH/hf}"
export HF_HUB_CACHE="$HF_HOME/hub"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/uv-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRATCH/triton}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TRITON_CACHE_DIR"

# -----------------------------------------------------------------------------
# Thread limits
#
# Without these, every worker spawns a full BLAS thread pool and the job
# thrashes. SLURM_CPUS_PER_TASK is the allocation, not the node's core count.
# -----------------------------------------------------------------------------
NCPUS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$NCPUS"
export MKL_NUM_THREADS="$NCPUS"
export TOKENIZERS_PARALLELISM=false
export JULIA_COPILOT_NUM_WORKERS="$NCPUS"   # see note in README / config.py

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
# SITE-SPECIFIC: most clusters need a CUDA module before torch will see the GPU.
# module purge
# module load cuda/12.4
# module load python/3.12

if ! command -v uv >/dev/null 2>&1; then
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { echo "uv not found; install it on the login node first." >&2; exit 1; }

# Compute nodes are frequently network-isolated. Sync on the login node before
# submitting; --offline here turns a silent hang into an immediate error.
echo "Syncing environment..."
uv sync --locked --extra train --offline || uv sync --locked --extra train

# -----------------------------------------------------------------------------
# Provenance -- these lines are what you will want when a run looks wrong
# -----------------------------------------------------------------------------
echo "=================================================================="
echo "Job              : ${SLURM_JOB_NAME:-local} (${SLURM_JOB_ID:-n/a})"
echo "Node             : $(hostname)"
echo "CPUs / Memory    : $NCPUS / ${SLURM_MEM_PER_NODE:-?} MB"
echo "CUDA_VISIBLE_DEV : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Model            : $MODEL"
echo "Data             : $DATA (frac=$FRAC, batch=$BATCH_SIZE, quantized=$QUANTIZED)"
echo "Commit           : $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
echo "Started          : $(date -Is)"
echo "=================================================================="
nvidia-smi || echo "nvidia-smi unavailable"

# -----------------------------------------------------------------------------
# Build the argument list
# -----------------------------------------------------------------------------
ARGS=(train
      --model "$MODEL"
      --data "$DATA"
      --frac-of-data "$FRAC"
      --batch-size "$BATCH_SIZE")

[[ "$QUANTIZED"    == "1" ]] && ARGS+=(--quantized)
[[ "$FIRST_LINE"   == "1" ]] && ARGS+=(--first-line)
[[ "$SIMPLE_INPUT" == "1" ]] && ARGS+=(--simple-input)

echo "Running: julia-copilot ${ARGS[*]}"

# srun so the step is accounted and signals propagate to the Python process.
srun --unbuffered uv run julia-copilot -v "${ARGS[@]}"

echo "=================================================================="
echo "Finished: $(date -Is)"
echo "Checkpoints under: checkpoints/${MODEL//\//_}/"
echo "=================================================================="
