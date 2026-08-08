#!/usr/bin/env bash
#SBATCH --job-name=coherax-e08-restarts
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --constraint=high_l3
#SBATCH --hint=nomultithread
#SBATCH --array=0-99%20
#SBATCH --output=cluster/logs/%x-%A-%a.out
#SBATCH --error=cluster/logs/%x-%A-%a.err

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
export COHERAX_HARDWARE_CONSTRAINT=high_l3
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

restart_count=100
pair_count=25
restart_index="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array}"
task_count="${SLURM_ARRAY_TASK_COUNT:?Missing Slurm array task count}"
if ((task_count != restart_count || restart_index < 0 || restart_index >= restart_count)); then
  echo "This matched benchmark requires exactly the array 0-99" >&2
  exit 2
fi

# One array task is one matched timing block. It runs every method/depth pair
# on the same allocation. Rotation balances cold-node/order effects: because
# 100 is divisible by 25, each pair occupies every execution position 4 times.
rotation=$((restart_index % pair_count))
for ((offset=0; offset<pair_count; offset++)); do
  pair_index=$(((rotation + offset) % pair_count))
  global_index=$((pair_index * restart_count + restart_index))
  srun --exclusive --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    --cpu-bind=cores \
    "$repo_root/scripts/safe_guard.sh" --wall-minutes 300 --memory-gb 7 -- \
      python cluster/_cluster_restart.py \
        --global-index "$global_index" \
        --restart-count "$restart_count"
done
