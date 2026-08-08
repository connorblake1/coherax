# Experiment 08: even cat-state preparation

## Question

This experiment compares five state representations while optimizing the same
echoed conditional displacement (ECD) circuit to prepare the even cat state

\[
|C_+(2)\rangle=
\frac{|-2\rangle+|2\rangle}{\sqrt{2+2e^{-8}}}.
\]

Here `|z>` is an oscillator coherent state centered at the complex
phase-space position `z`. The two components are centered at the real points
`-2` and `+2`. Coherent states are not orthogonal:

\[
\langle-2|2\rangle=e^{-8}.
\]

That overlap explains the normalization `sqrt(2 + 2 exp(-8))`. This is the
even cat: only even photon numbers have nonzero target amplitudes. Its exact
mean photon number is `4 tanh(4)`, approximately `3.9973`.

The protocol uses depths 4, 8, 12, 16, and 20; three deterministic restarts;
10,000 Adam updates per restart; the
same initial circuit at a fixed depth for every method; the same five state
representations; exact final rescoring; the same four line plots; and the same
five-by-five Wigner-function layout plus an exact target panel.

## Initial state, circuit, and random initialization

Every circuit begins in qubit ground state and oscillator vacuum,

\[
|\Psi_0\rangle=|0\rangle_q\otimes|0\rangle_o.
\]

One saved layer is

```text
[beta_real, beta_imag, phi, theta, gamma = 0]
```

with complex ECD displacement `beta = beta_real + i beta_imag`. After the
qubit rotation, the two output branches receive oscillator displacements
`-beta/2` and `+beta/2`. Angles are radians.

At every layer and restart,

\[
\beta\sim\mathcal{CN}(0,0.5),\qquad
\operatorname{Re}\beta,\operatorname{Im}\beta
\stackrel{\rm iid}{\sim}\mathcal N(0,0.25),
\]

so `E[|beta|^2] = 0.5`. Here `iid` means independent and identically
distributed. `phi` and `theta` are independent uniform draws on `[0, 2 pi]`;
`gamma` is exactly zero. Adam is an adaptive gradient optimizer. Every
restart receives 10,000 Adam updates.

## Exact common score

The analytic circuit output is

\[
|\Psi\rangle=\sum_{q=0}^1|q\rangle\sum_j a_{qj}|z_{qj}\rangle.
\]

The final score traces out the qubit:

\[
F_{\rm exact}=\sum_{q=0}^1
\left|\sum_j a_{qj}\langle C_+(2)|z_{qj}\rangle\right|^2,
\]

where

\[
\langle C_+(2)|z\rangle=
\frac{\langle-2|z\rangle+\langle2|z\rangle}{\sqrt{2+2e^{-8}}},
\]

\[
\langle d|z\rangle=\exp\!\left[-\frac{|d-z|^2}{2}
+i(\operatorname{Re}d\operatorname{Im}z-
\operatorname{Im}d\operatorname{Re}z)\right].
\]

This score has no Fock cutoff and no packet compression. `1 - F_exact` is the
exact infidelity. Within a method and depth, the restart with largest
`F_exact` is saved. “Exact” means exact for this explicitly defined
two-component target up to float64 arithmetic; it does not mean exact hardware
execution.

All methods train against the normalized projection onto photon numbers 0
through 47. For this mean-photon-number-four target, the retained norm rounds
to `1.0` in float64. The projection is therefore numerically converged, but
the cutoff-free rescore is still used uniformly.

## Five methods

### Full coherent paths

Displacement of a coherent packet is analytic:

\[
c|z\rangle\mapsto
c\,e^{-i\Omega(s,z)}|z+s\rangle,
\qquad
\Omega(a,b)=\operatorname{Re}a\operatorname{Im}b-
\operatorname{Im}a\operatorname{Re}b.
\]

The qubit rotation doubles the path count each layer, so depth `n` has `2^n`
paths and no oscillator cutoff.

### Analytic Fock recurrence

The Fock basis is the photon-number basis `|0>, |1>, ...`. This method carries
two 48-entry arrays and constructs the upper-left block of the exact infinite
displacement using

\[
D(a)|0\rangle=|a\rangle,\qquad
D(a)|r\rangle=
\frac{(\hat a^\dagger-a^*)D(a)|r-1\rangle}{\sqrt r}.
\]

The retained block is not necessarily unitary because amplitude can leave the
48-level grid.

### Dense numerical Fock

This method truncates the annihilation operator first,

\[
a_{48}=\sum_{r=1}^{47}\sqrt r,|r-1\rangle\langle r|,
\qquad
D^{\rm num}_{48}(a)=\exp(a a_{48}^\dagger-a^*a_{48}),
\]

then materializes each 96-by-96 joint layer unitary

\[
U_j=
\begin{pmatrix}
D^{\rm num}_{48}(-\beta_j/2)&0\\
0&D^{\rm num}_{48}(+\beta_j/2)
\end{pmatrix}(R_j\otimes I_{48}).
\]

This finite matrix is unitary, but its boundary is an artifact of truncating
before exponentiation. Matrix exponentials and joint-state propagation use
complex64; the common exact rescore uses float64.

### Continuous anchors

This online compressor retains 32 coherent packets per qubit branch. It also
carries two 48-entry working Fock arrays. When a branch has more than 32
packets, it chooses residual pivots, takes 12 inner Adam steps that move the
complex centers, and globally refits coefficients:

\[
x=(B^\dagger B+10^{-9}I)^{-1}B^\dagger v,
\qquad |\widetilde v\rangle=Bx.
\]

Between layers it carries 64 packet coefficients plus 96 Fock amplitudes. Just
before compression it can carry 128 packet candidates plus 96 amplitudes, so
the propagated state remains below 1,024 coefficients.

### Squeezed packets

This uses the same movable centers and coefficient fit, then takes 16 more
inner Adam steps to optimize one bounded complex squeeze per packet. Both
packet methods use a straight-through outer derivative: the forward pass uses
the compressed state, but the derivative of discrete packet selection is
approximated by the identity. This is a heuristic.

## Results

Wall time includes all three restarts, first compilation, and exact host
rescoring. Native fidelity means the method's internal 48-level score;
headline results and winner selection use only `F_exact`.

<!-- RESULTS_TABLE -->

<!-- RESULTS_DISCUSSION -->

## Gate-time formulas

The Coherax estimate is

\[
T_C=\sum_j\left[24\,{\rm ns}+\max\!\left(
\frac{|\beta_j|}{(2\pi\,50\,{\rm kHz})20},48\,{\rm ns}
\right)\right].
\]

The separate Eickbusch-hardware curve is the explicit lower bound

\[
T_{E,{\rm LB}}=\sum_j\max\!\left(
\frac{|\beta_j|}{(2\pi\,33\,{\rm kHz})30},224\,{\rm ns}
\right).
\]

It is not an upstream compiled waveform duration. Search wall time and
hardware execution time are different quantities.

## Files and reproduction

- `cat_state_preparation_optimizers.ipynb`: self-contained executed notebook.
- `results.json`: scores, timers, configuration, and validations.
- `best_sequences.npz` and `sequences/`: all winning physical circuits.
- `figs/`: wall-time, infidelity, two gate-time plots, and Wigner plate.
- `report.pdf`: short self-contained report.
- `workflow_cat_state_preparation_optimizers.sh`: one-command reproduction.

Run from this directory:

```bash
./workflow_cat_state_preparation_optimizers.sh
```

<!-- RUNTIME_NOTE -->

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
