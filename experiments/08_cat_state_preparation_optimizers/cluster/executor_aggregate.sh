#!/usr/bin/env bash
#SBATCH --job-name=coherax-e08-aggregate
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --error=cluster/logs/%x-%j.err

set -euo pipefail

experiment_dir="${SLURM_SUBMIT_DIR:?Submit this script from its experiment directory}"
cluster_dir="$experiment_dir/cluster"
repo_root="$(cd "$experiment_dir/../.." && pwd)"
cd "$experiment_dir"

module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${COHERAX_CONDA_ENV:-coherax}"

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

restart_count=100
"$repo_root/scripts/safe_guard.sh" --wall-minutes 210 --memory-gb 11 -- \
  python cluster/_aggregate_cluster.py \
    --restart-count "$restart_count" \
    --publish
