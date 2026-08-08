#!/usr/bin/env bash
set -euo pipefail

cluster_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd "$cluster_dir/.." && pwd)"
cd "$experiment_dir"

restart_count=100
worker_count=100

if (( $# != 0 )); then
  echo "usage: $0" >&2
  exit 2
fi

if [[ ! -d "$cluster_dir" ]]; then
  echo "submit_cluster.sh: expected cluster directory is missing: $cluster_dir" >&2
  exit 2
fi
for runtime_directory in "$cluster_dir/logs" "$cluster_dir/results"; do
  if [[ ! -d "$runtime_directory" ]]; then
    if ! mkdir -- "$runtime_directory"; then
      echo "submit_cluster.sh: cannot create $runtime_directory" >&2
      ls -ld "$experiment_dir" "$cluster_dir" >&2 || true
      exit 2
    fi
  fi
  if [[ ! -w "$runtime_directory" ]]; then
    echo "submit_cluster.sh: directory is not writable: $runtime_directory" >&2
    ls -ld "$experiment_dir" "$cluster_dir" "$runtime_directory" >&2 || true
    echo "Use a clone owned by your Engaging account." >&2
    exit 2
  fi
done
array_spec="0-$((worker_count - 1))%1"
array_job_id="$(
  sbatch --parsable \
    --array="$array_spec" \
    --output="$cluster_dir/logs/%x-%A-%a.out" \
    --error="$cluster_dir/logs/%x-%A-%a.err" \
    cluster/executor.sh
)"
aggregate_job_id="$(
  sbatch --parsable \
    --dependency="afterok:$array_job_id" \
    --output="$cluster_dir/logs/%x-%j.out" \
    --error="$cluster_dir/logs/%x-%j.err" \
    cluster/executor_aggregate.sh
)"

echo "restart array job: $array_job_id"
echo "dependent aggregate job: $aggregate_job_id"
echo "design: 100 matched workers x 25 cases = 2500 optimizations"
echo "resource cap: one array task (one node, four CPU cores) at a time"
echo "monitor with: squeue --me"
