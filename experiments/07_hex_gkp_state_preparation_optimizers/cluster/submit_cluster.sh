#!/usr/bin/env bash
set -euo pipefail

cluster_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd "$cluster_dir/.." && pwd)"
cd "$experiment_dir"

restart_count=100
worker_count=100
max_concurrent="${1:-20}"

if ((max_concurrent < 1 || max_concurrent > 24)); then
  echo "usage: $0 [maximum_concurrent_workers=1..24]" >&2
  exit 2
fi

mkdir -p cluster/logs cluster/results
array_spec="0-$((worker_count - 1))%$max_concurrent"
array_job_id="$(
  sbatch --parsable \
    --array="$array_spec" \
    cluster/executor.sh
)"
aggregate_job_id="$(
  sbatch --parsable \
    --dependency="afterok:$array_job_id" \
    cluster/executor_aggregate.sh
)"

echo "restart array job: $array_job_id"
echo "dependent aggregate job: $aggregate_job_id"
echo "design: 100 matched workers x 25 cases = 2500 optimizations"
echo "monitor with: squeue --me"
