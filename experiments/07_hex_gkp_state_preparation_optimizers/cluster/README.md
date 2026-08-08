# MIT Engaging parallel restarts

These scripts run each `(method, depth, restart)` optimization independently
on the Engaging Slurm scheduler, then select winners by cutoff-free exact
fidelity and regenerate the experiment artifacts.

The command requests exactly 100 restarts for every one of the 25 method/depth
pairs, for 2,500 independent optimizations:

```bash
cd experiments/07_hex_gkp_state_preparation_optimizers
cluster/submit_cluster.sh
```

The launcher takes no arguments. The 100 array tasks correspond exactly to
restart indices 0 through 99. Each task
runs all 25 method/depth cases as fresh Python processes on one allocation.
Thus every method/depth pair is timed on the same multiset of 100 allocated
nodes. At a fixed depth and restart index, every method also receives the same
deterministic initial circuit. The 25-case execution order is rotated by
restart index, so every case occupies every order position four times. The
`%1` array cap makes the 100 workers run strictly sequentially. At most one
node, one Slurm task, and four CPU cores are active at a time.

Before submission, create or copy the repository on Engaging and make sure the
`coherax` Conda environment contains the repository dependencies. One initial
setup from the repository root is:

```bash
module load miniforge/24.3.0-0
conda create -n coherax python=3.11 pip -y
conda activate coherax
python -m pip install -e .
```

To use an already-created environment with a different name:

```bash
export COHERAX_CONDA_ENV=my_environment
cluster/submit_cluster.sh
```

`executor.sh` uses the CPU-only `mit_normal` partition, four CPUs, 8 GB per
array task, and a twelve-hour task limit. It requests the Engaging `high_l3`
constraint so all timing tasks use the same documented AMD EPYC 9384X CPU
model. Thread counts, CPU-only JAX, float64 mode, environment, memory, and CPU
requests are identical for every case, and simultaneous multithreading is
disabled for the allocation. Each optimizer is launched as a core-bound
four-CPU Slurm job step. `executor_aggregate.sh` is submitted automatically
with an `afterok` dependency; it runs only after the whole array succeeds.
Restart shards are atomic and resumable: resubmitting skips completed JSON/NPZ
pairs. Delete a specific pair or pass `--force` directly to
`_cluster_restart.py` only when that restart really should be recomputed.

The aggregator writes `aggregate_results.json`, `aggregate_summary.csv`, and
`aggregate_best_sequences.npz` in this directory. It also publishes the
winning sequences, canonical `results.json`, plots, and Wigner plate. In a
parallel run, the plotted multirestart time is the **sum of independent
restart wall times**, not the scheduler latency between submission and final
completion; this distinction is recorded in the JSON. Before publishing, the
aggregator verifies that every restart ran all 25 cases on one physical node
and that all 100 workers report the same CPU, resource, and numerical-software
signature. It fails
loudly instead of publishing a hardware-mismatched timing comparison.

The matching guarantee applies to the measured optimizer processes, not to
queue wait or aggregation time. If a partially completed restart is resumed
on another node, the hardware validator will reject it; rerun all 25 shards
for that restart as one block.

Useful Engaging commands:

```bash
squeue --me
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
sinfo -p mit_normal
```
