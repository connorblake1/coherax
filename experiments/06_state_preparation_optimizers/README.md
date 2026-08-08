# Experiment 06: end-to-end state-preparation optimizers

## Question

Can an online compressed wavefunction representation improve end-to-end echoed
conditional displacement (ECD) circuit optimization for the oscillator target

\[
|\psi_{\mathrm{target}}\rangle
=\frac{|9\rangle+|10\rangle}{\sqrt{2}}?
\]

Here `|9>` means the oscillator number state with exactly nine photons, and
`|10>` means exactly ten photons. The initial joint state is qubit ground state
times oscillator vacuum, `|0>_q |0>_o`. The tested circuit depths are 4, 8, 12,
16, and 20 ECD-plus-qubit-rotation layers.

The answer in this benchmark is **no**. At 20 layers, all five optimizers find
almost the same physical circuit quality, with exact fidelity near 0.998765.
The analytic Fock-recurrence optimizer is by far the fastest. The full analytic optimizer is
cutoff-free and gives the largest exact fidelity, but only by
`8.43e-9` over the dense numerical Fock method. The two online packet methods
are much slower and do not improve the final fidelity.

## Important optimization-budget note

Adam is an adaptive gradient optimizer. The original depths 4, 8, 12, and 16
used 1,200 Adam updates per restart. The
new depth-20 runs use 10,000 updates per restart because 1,200 updates can stop
before a late optimization breakthrough. Every method still uses the same
three deterministic restarts at any fixed depth.

Consequently, comparisons **between methods at depth 20 are budget-matched**.
The change from depth 16 to depth 20 is **not a controlled depth-only
comparison**: it changes both the number of circuit layers and the number of
Adam updates. The lower depth-20 infidelity therefore cannot be attributed to
the four extra layers alone. A fully controlled depth-scaling study would need
10,000 updates at every depth.

## What one saved layer means

Every layer is saved as

```text
[beta_real, beta_imag, phi, theta, gamma = 0]
```

The complex ECD amplitude is `beta = beta_real + i beta_imag`. The angles are
in radians. In this repository's ECD convention, the two oscillator branches
are displaced by `-beta/2` and `+beta/2` after the qubit rotation.

The layer JSON files in [`sequences/`](sequences/) spell out every field; the
same arrays are collected in [`best_sequences.npz`](best_sequences.npz).

## The five optimizer labels

All methods optimize the physical layer parameters from end to end. They differ
only in how they calculate the state and objective during the search.

1. **Full coherent paths** keeps all analytic coherent-state paths. A depth
   `n` objective evaluation has `2^n` paths, so the depth-20 calculation has
   1,048,576 path coefficients. It has no oscillator Fock cutoff.

2. **Analytic Fock recurrence** propagates the two qubit branches as two arrays of 48
   Fock coefficients. The Fock basis is the photon-number basis
   `|0>, |1>, |2>, ...`, so each array stores photon numbers 0 through 47.
   Displacement matrices are constructed with the exact
   recurrence

   \[
   D(a)|0\rangle=|a\rangle,\qquad
   D(a)|k\rangle=
   \frac{(\hat a^\dagger-a^*)D(a)|k-1\rangle}{\sqrt{k}}.
   \]

   This is the same basic dense Fock-grid optimization strategy used for ECD
   state preparation by Eickbusch *et al.* The implementation and target in
   this experiment are repository-native.

3. **Dense numerical Fock** first truncates the annihilation operator,

   \[
   a_{48}=\sum_{r=1}^{47}\sqrt r\,|r-1\rangle\langle r|,
   \qquad D_{48}^{\rm num}(a)=
   \exp(a a_{48}^\dagger-a^*a_{48}),
   \]

   then materializes each full `96 x 96` qubit-oscillator layer unitary. This
   finite displacement is unitary inside the cutoff, but its boundary differs
   from the upper-left block of the exact infinite-dimensional displacement.
   Matrix exponentials and joint-state propagation use complex64; the target
   overlap can promote precision. Final rescoring uses the common float64
   cutoff-free formula.

4. **Squeezed packets** retains 32 packets in each qubit branch, or 64 total.
   After a layer creates more than 32 candidates per branch, greedy residual
   pivots choose seeds, 12 inner Adam steps move their complex centers, a
   global least-squares solve refits their coefficients, and 16 more inner
   steps optimize one bounded complex squeeze per packet.

5. **Continuous anchors** uses the same 32 movable centers per branch and the
   same global coefficient refit, but stops before the squeeze optimization.

The packet methods use a 48-level working Fock vector to define the local
projection, but they retain only 64 packet coefficients between layers. Their
propagated data therefore contains 160 complex coefficients between layers and
at most 224 immediately before 128 packet candidates are compressed.
Their
forward objective uses the projected state. Their outer gradient uses a
straight-through estimator: the derivative of the discrete projection is
approximated by the identity. This is a heuristic, so it is not used for final
ranking.

## Exact fidelity and winner selection

The exact output is written as

\[
|\Psi\rangle=\sum_{q=0}^1|q\rangle
\sum_j c_{qj}|z_{qj}\rangle.
\]

The qubit is traced out. The exact oscillator fidelity is

\[
F=\sum_{q=0}^1\left|
\sum_j c_{qj}e^{-|z_{qj}|^2/2}
\frac{z_{qj}^9/\sqrt{9!}+z_{qj}^{10}/\sqrt{10!}}{\sqrt2}
\right|^2.
\]

This score is calculated without a Fock cutoff. The saved winner is the
restart with the largest value of this common exact `F`, never the largest
internal approximate fidelity. `1 - F` is the exact infidelity.

An independent conversion with 96 Fock levels agrees with every saved exact
fidelity to within `7.1e-13`. The larger depth-20 value is still numerical
roundoff, not a visible physical discrepancy.

## Results

The wall time is the total time for all three restarts at one method and depth,
including the first JAX compilation and the cutoff-free host rescore after each
restart. It is the actual multirestart cost paid to obtain the displayed winner.

| method | layers | Adam steps/restart | exact fidelity `F` | exact `1-F` | total wall time (s) |
|---|---:|---:|---:|---:|---:|
| full coherent paths | 4 | 1,200 | 0.505018513 | 0.494981487 | 2.50 |
| analytic Fock recurrence | 4 | 1,200 | 0.505018513 | 0.494981487 | 3.74 |
| dense numerical Fock | 4 | 1,200 | 0.505018552 | 0.494981448 | 17.44 |
| squeezed packets | 4 | 1,200 | 0.505018513 | 0.494981487 | 4.24 |
| continuous anchors | 4 | 1,200 | 0.505018513 | 0.494981487 | 5.04 |
| full coherent paths | 8 | 1,200 | 0.857536427 | 0.142463573 | 5.87 |
| analytic Fock recurrence | 8 | 1,200 | 0.857534952 | 0.142465048 | 7.21 |
| dense numerical Fock | 8 | 1,200 | 0.857536425 | 0.142463575 | 33.35 |
| squeezed packets | 8 | 1,200 | 0.857534965 | 0.142465035 | 114.67 |
| continuous anchors | 8 | 1,200 | 0.857534973 | 0.142465027 | 63.75 |
| full coherent paths | 12 | 1,200 | 0.974346473 | 0.025653527 | 14.53 |
| analytic Fock recurrence | 12 | 1,200 | 0.974350445 | 0.025649555 | 10.87 |
| dense numerical Fock | 12 | 1,200 | 0.974345335 | 0.025654665 | 56.51 |
| squeezed packets | 12 | 1,200 | 0.974348265 | 0.025651735 | 360.45 |
| continuous anchors | 12 | 1,200 | 0.974351803 | 0.025648197 | 180.37 |
| full coherent paths | 16 | 1,200 | 0.988280951 | 0.011719049 | 36.70 |
| analytic Fock recurrence | 16 | 1,200 | 0.986552032 | 0.013447968 | 14.77 |
| dense numerical Fock | 16 | 1,200 | 0.988280877 | 0.011719123 | 77.49 |
| squeezed packets | 16 | 1,200 | 0.987050831 | 0.012949169 | 659.39 |
| continuous anchors | 16 | 1,200 | 0.987752877 | 0.012247123 | 297.66 |
| full coherent paths | 20 | 10,000 | **0.998767876** | **0.001232124** | 2,038.69 |
| analytic Fock recurrence | 20 | 10,000 | 0.998767507 | 0.001232493 | **53.22** |
| dense numerical Fock | 20 | 10,000 | 0.998767867 | 0.001232133 | 483.39 |
| squeezed packets | 20 | 10,000 | 0.998764935 | 0.001235065 | 4,703.63 |
| continuous anchors | 20 | 10,000 | 0.998764050 | 0.001235950 | 2,959.72 |

At four layers no packet compression occurs: each qubit branch has only eight
paths, below the retained budget of 32. The full, analytic-Fock, and packet
methods therefore share the same native dynamics at this depth; dense Fock
still uses a different truncated matrix exponential. At depth 20, all five
winning runs are restart 1, the same warm start. The third, random restart remains much worse even after
10,000 updates; for example, its full-coherent exact fidelity is 0.984637.
This seed dependence supports the concern that short Adam runs can miss late
breakthroughs.

At depth 20, analytic Fock is 38.3 times faster than full coherent paths,
9.1 times faster than dense numerical Fock, 88.4 times faster than squeezed
packets, and 55.6 times faster than continuous anchors. Dense numerical Fock is
4.2 times faster than full coherent paths and differs from its exact fidelity
by only `8.43e-9`.
The packet native objectives overestimate their exact fidelities by
`6.84e-5` (squeezed) and `1.00e-4` (continuous), which is why exact rescoring
matters.

## Gate-time definitions

Optimization wall time is computer search time. Gate time is the estimated
duration of running the saved circuit on hardware. They are different units
and should not be compared directly.

The Coherax timing convention is

\[
T_C=\sum_i\left[24\,\mathrm{ns}+
\max\left(\frac{|\beta_i|}{(2\pi\,50\,\mathrm{kHz})20},
48\,\mathrm{ns}\right)\right].
\]

The repository does not contain the upstream Eickbusch pulse compiler for new
arbitrary sequences. The separate Eickbusch plot therefore uses the explicit
layerwise hardware lower bound

\[
T_{E,\mathrm{LB}}=\sum_i\max\left(
\frac{|\beta_i|}{(2\pi\,33\,\mathrm{kHz})30},224\,\mathrm{ns}\right).
\]

It is **not** a compiled waveform duration. The 224 ns floor is two 24 ns
qubit pulses plus four 44 ns displacement pulses.

At depth 20, the five `T_C` values lie between 3.880 and 3.883 microseconds.
The five Eickbusch lower bounds lie between 4.893 and 4.896 microseconds. The
optimizer representation changes search cost much more than it changes the
gate time of the winning physical sequence.

## Wigner functions

[`figs/wigner_optimizer_grid.png`](figs/wigner_optimizer_grid.png) was generated
with `dq.plot.wigner`. From top to bottom, the optimizer rows are full coherent
paths, analytic Fock recurrence, dense numerical Fock, squeezed packets, and
continuous anchors; columns are the five layer counts. The single bottom panel
is the exact target. Every optimizer panel is
reconstructed from the exact physical circuit and then traced over the qubit;
it is not a plot of an internal compressed surrogate.

## Honest takeaway

For this target and matched depth-20 optimization budget, online packet
compression has no practical advantage over the analytic Fock recurrence. It also has no
accuracy advantage over the uncompressed analytic optimizer. Its real benefit
is a bounded packet count that does not grow as `2^n`, but the repeated
center/squeeze fitting makes it the slowest choice here.

The analytic Fock recurrence is the best speed choice because the optimized
displacements stay in a regime where 48 levels are sufficient. The full
analytic representation retains the stronger guarantee: its objective remains
exact for arbitrarily large displacements and has no truncation boundary. That
robustness is not needed by these winning sequences, but it remains a real
advantage outside this benchmark.

The dense numerical result shows that truncating before exponentiating is
accurate for these particular sequences, but its matrix-exponential gradients
cost about nine times more than the analytic recurrence. It does not improve
the best fidelity.

All five methods exceed `F = 0.99876` at depth 20 under the three-restart,
10,000-step budget. These are best-observed values, not proofs of the globally
optimal circuit at any depth.

The closeness of those scores must not be overinterpreted. Only three
deterministic restarts were run for each method/depth pair. Moreover, all five
methods selected the same winning restart at each depth: restart 1 at depths
4, 8, 12, and 20, and restart 2 at depth 16. A fixed `(depth, restart)` used
the same initial circuit for every method. The searches were therefore highly
correlated; this run is too small to establish optimizer equivalence.

## Files

- [`state_preparation_optimizers.ipynb`](state_preparation_optimizers.ipynb):
  self-contained formulas, code, execution, and artifact generation.
- [`results.json`](results.json): all restart scores, timers, winners, timing
  values, validation records, and environment information.
- [`best_sequences.npz`](best_sequences.npz): all 25 winning arrays.
- [`sequences/`](sequences/): one readable JSON layer sequence per winner.
- [`figs/`](figs/): the four requested line plots and Wigner-function plate.
- [`report.pdf`](report.pdf): short self-contained report.
- [`workflow_state_preparation_optimizers.sh`](workflow_state_preparation_optimizers.sh):
  one-command reproduction under `scripts/safe_guard.sh`.

## Reproduce

From this directory:

```bash
./workflow_state_preparation_optimizers.sh
```

The recorded JAX CPU runs took about 30 minutes for depths 4--16 and 2 hours
43 minutes for the depth-20 extension; the dense numerical benchmark adds
about 11 minutes. A clean complete rerun should therefore be budgeted for
roughly 3.5 hours on the same machine.

## Remove old generated results

The cleanup script uses an explicit allowlist: it removes numerical result
files, figures, saved sequences, compiled-report files, caches, cluster
shards, and cluster logs. It preserves source code, shell workflows,
documentation, `report.tex`, and the notebook source; notebook execution
counts and outputs are cleared in place.

```bash
./clean.sh --dry-run  # list what would be removed
./clean.sh            # perform the cleanup
```

Do not run it when you intend to resume an incomplete cluster sweep, because
cluster restart shards are deliberately included in the cleanup.

## Many restarts on MIT Engaging

The [`cluster/`](cluster/) directory contains an Engaging Slurm job-array
launcher for exactly 100 restarts per method and layer count (2,500
optimizations). Each of the 100 hardware-matched workers runs all 25 cases as
fresh processes on one allocation and writes atomic JSON/NPZ shards. Launch
with at most 20 workers running at once using:

```bash
cluster/submit_cluster.sh 20
```

The dependent aggregation job selects each winner by exact analytic fidelity
and regenerates the experiment artifacts. See [`cluster/README.md`](cluster/README.md)
for the homogeneous CPU constraint, matched timing design, validation,
environment setup, monitoring, and safe resumption.
