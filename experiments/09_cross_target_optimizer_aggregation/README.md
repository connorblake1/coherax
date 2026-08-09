# Experiment 09: aggregate comparison across three target states

## Purpose

This experiment does **not** optimize any new circuits. It combines the complete
100-restart cluster results from Experiments 06, 07, and 08 into one validated
comparison. The three targets are:

1. the Fock-state superposition
   \((|9\rangle+|10\rangle)/\sqrt{2}\);
2. the finite hexagonal GKP logical-zero state with envelope
   \(\Delta=0.2\) and shell radius 14;
3. the even cat state
   \((|-2\rangle+|2\rangle)/\sqrt{2+2e^{-8}}\).

For each target, the inputs contain five circuit-simulation or compression
methods, five circuit depths \(n\in\{4,8,12,16,20\}\), and 100 optimizer
restarts. Therefore the complete input contains

\[
3\text{ targets}\times5\text{ methods}\times5\text{ depths}
\times100\text{ restarts}=7{,}500
\]

separately timed optimizer-run records. At a fixed target, depth, and restart
index, the five methods intentionally share the same initial circuit, so those
five records are paired rather than statistically independent. Experiment 09
refuses to run if any record is missing, duplicated, internally inconsistent,
or taken from a source aggregate whose declared restart count is not exactly
100.

## Required inputs

After copying or pulling the cluster results into the checkout, these files must
exist:

```text
experiments/06_state_preparation_optimizers/cluster/aggregate_results.json
experiments/07_hex_gkp_state_preparation_optimizers/cluster/aggregate_results.json
experiments/08_cat_state_preparation_optimizers/cluster/aggregate_results.json
```

Each source cluster aggregator must have been run with `--restart-count 100`.
The source files are read-only inputs: Experiment 09 never changes Experiments
06–08.

## What the five methods mean

Every method optimizes the same physical echoed conditional displacement (ECD)
circuit at a fixed target and depth. Only the state representation used during
optimization changes.

- **Full coherent paths:** retains every analytic coherent-state path. A depth
  \(n\) circuit has \(2^n\) paths and no oscillator cutoff.
- **Analytic Fock recurrence:** retains two 48-entry Fock vectors and constructs
  the upper-left block of the infinite-dimensional displacement operator using

  \[
  D(a)|0\rangle=|a\rangle,\qquad
  D(a)|r\rangle=
  \frac{(\hat a^\dagger-a^*)D(a)|r-1\rangle}{\sqrt r}.
  \]

- **Dense numerical Fock:** first truncates the annihilation operator to 48
  levels, then materializes

  \[
  D_{48}^{\mathrm{num}}(a)=
  \exp(a\hat a_{48}^\dagger-a^*\hat a_{48})
  \]

  and the full 96-by-96 qubit–oscillator layer unitary.
- **Continuous anchors:** retains 32 movable coherent packets per qubit branch.
  Candidate centers are chosen from residuals, moved by inner Adam updates, and
  globally refit by

  \[
  x=(B^\dagger B+10^{-9}I)^{-1}B^\dagger v.
  \]

- **Squeezed packets:** uses the same movable centers and coefficient fit, plus
  one optimized complex squeeze per retained packet.

The two packet methods compress after every circuit layer and never build the
full depth-20 path tree. Their discrete selection uses a straight-through
gradient approximation.

## Exact fidelity and winner selection

Let the exact joint circuit output be

\[
|\Psi\rangle=\sum_{q=0}^{1}|q\rangle|\psi_q\rangle,
\]

where \(q\) is the final qubit value. For normalized oscillator target
\(|\tau\rangle\), every saved circuit is rescored by

\[
F_{t,m,n,r}=\sum_{q=0}^{1}
|\langle\tau_t|\psi_{q,m,n,r}\rangle|^2.
\]

The qubit is traced out, which explains the sum. This final score is evaluated
from the physical ECD sequence with analytic coherent-state overlaps. It does
not use a Fock cutoff or the packet approximation. “Exact” means exact for the
stated ideal circuit and target, up to floating-point arithmetic; it does not
include hardware noise.

For target \(t\), method \(m\), and depth \(n\), the reported winner is

\[
F^*_{t,m,n}=\max_{r=0,\ldots,99}F_{t,m,n,r},
\qquad
\epsilon^*_{t,m,n}=1-F^*_{t,m,n}.
\]

No method's internal training loss is used to select the winner.

## Restart reliability

A best-of-100 number does not show how often an optimizer succeeds. Experiment
09 therefore also reports the median and interquartile range over restarts and
the success probability at thresholds \(F\ge0.9\), \(F\ge0.99\), and
\(F\ge0.999\):

\[
\widehat p_\tau=
\frac{1}{100}\sum_{r=0}^{99}\mathbf 1[F_{t,m,n,r}\ge\tau].
\]

The success-rate plot uses 95 percent Wilson binomial intervals. It also records
a restart win share: at fixed target, depth, and restart index, the method with
largest exact fidelity receives one vote; exact ties split that vote.

## Wall time and fair speed comparisons

For one method, depth, and target, total search time is

\[
C_{t,m,n}=\sum_{r=0}^{99}T_{t,m,n,r},
\]

where one \(T\) includes optimization and exact rescoring. This is the sum of
100 process wall times. It is **not** Slurm queue delay and is **not** elapsed
calendar time for a parallel array.

Different restart blocks can land on different node models. Raw search time is
shown separately for each target and retains that limitation. The cleaner
within-block comparison is the paired speedup

\[
S_{t,m,n,r}=\frac{T_{t,\mathrm{full},n,r}}{T_{t,m,n,r}}.
\]

It compares the same target, depth, restart index, and node allocation.
Experiment 09 excludes any restart block that the source aggregator marked as
hardware-mismatched, then plots the median \(S\) and its interquartile range.
Values above one mean the tested method was faster than full coherent paths.

## Physical gate times

Optimizer wall time is computer search cost. It is separate from the time to
execute the selected circuit on hardware. The two existing estimates are
carried into the aggregate without recomputation:

\[
T_C=\sum_j\left[24\,{\rm ns}+\max\!\left(
\frac{|\beta_j|}{(2\pi\,50\,{\rm kHz})20},48\,{\rm ns}\right)\right],
\]

\[
T_{E,{\rm LB}}=\sum_j\max\!\left(
\frac{|\beta_j|}{(2\pi\,33\,{\rm kHz})30},224\,{\rm ns}\right).
\]

The second quantity is a lower bound using Eickbusch hardware parameters, not a
compiled pulse duration.

## Outputs

- `cross_target_optimizer_aggregation.ipynb`: self-contained executed notebook;
- `results.json`: definitions, input hashes, validation, 75 winner summaries,
  and five cross-task method summaries;
- `all_restart_results.csv`: all 7,500 restart-level measurements;
- `winner_summary.csv`: one row per target/method/depth;
- `method_summary.csv`: descriptive comparison across all 15 target/depth tasks;
- `source_artifact_inventory.csv`: records which optional sequences, hardware
  tables, and Wigner grids were copied back;
- `figs/`: exact infidelity, restart reliability, raw search time, paired
  speedup, two physical gate-time plots, and a fidelity heatmap;
- `report.pdf`: self-contained written report.

The geometric mean infidelity in `method_summary.csv` is only a compact
descriptive statistic across different targets. It is not itself a physical
state fidelity and should not replace the target-by-target plots.

## Run after the cluster data are present

From this directory:

```bash
./workflow_cross_target_optimizer_aggregation.sh
```

To check inputs and generate tables without plots:

```bash
python _aggregate_results.py --skip-plots
```

If an input is absent or incomplete, the command stops with the exact missing
path or inconsistent method/depth pair.

## Remove Experiment 09 outputs

The cleanup script only removes artifacts generated inside Experiment 09. It
never removes the source results in Experiments 06–08.

```bash
./clean.sh --dry-run
./clean.sh
```
