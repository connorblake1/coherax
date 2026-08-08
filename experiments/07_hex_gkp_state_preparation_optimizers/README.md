# Experiment 07: finite hex-GKP state preparation

## Question and headline result

This experiment asks how five state representations affect end-to-end
optimization of the same echoed conditional displacement (ECD) circuit for a
finite-energy hexagonal Gottesman--Kitaev--Preskill (GKP) state.
The target is the logical-zero state with envelope parameter `Delta = 0.2`.
The tested circuit depths are 4, 8, 12, 16, and 20 layers.

The five labels mean:

1. **full coherent paths:** retain every analytic coherent-state path;
2. **analytic Fock recurrence:** retain 48 Fock coefficients and evaluate the
   exact infinite-dimensional displacement matrix elements by a recurrence;
3. **dense numerical Fock:** truncate the ladder operator to 48 levels, take a
   dense matrix exponential, and materialize every 96-by-96 joint layer unitary;
4. **squeezed packets:** retain 32 movable squeezed packets per qubit branch;
5. **continuous anchors:** retain 32 movable coherent packets per qubit branch.

`F` below always means the common exact final fidelity, not an optimizer's
internal 48-level score. The complete numerical results are given after the
definitions.

## Exact definition of the target

This report uses the logical-zero state, written `|0_L>`. Define

\[
\alpha=\sqrt{\pi/\sqrt3},\qquad
\eta=e^{2\pi i/3}\alpha .
\]

The second lattice vector is called `eta` here so it is not confused with a
circuit displacement, which is conventionally called `beta`. A lattice site
has integer coordinates `(m,l)` and complex phase-space position

\[
d_{ml}=m\alpha+l\eta.
\]

The finite target retains every site in the complete hexagonal shell

\[
\max(|m|,|l|,|m-l|)\leq 14
\]

whose first coordinate `m` is even. Put `k=m/2`. Before normalization, the
coefficient of the coherent state `|d_ml>` is

\[
\widetilde c_{ml}
=\exp\!\left[-i\pi kl-\Delta^2|d_{ml}|^2\right],
\qquad \Delta=0.2.
\]

Thus the target used for every reported exact score is

\[
|G\rangle=\frac{1}{\sqrt{\mathcal N}}
\sum_{\substack{m,l\in\mathbb Z\\
\max(|m|,|l|,|m-l|)\leq14\\m\ {m even}}}
\widetilde c_{ml}|d_{ml}\rangle.
\]

There are 323 coherent sites. The normalization is not the Euclidean norm of
the 323 coefficients because coherent states are not orthogonal. It is

\[
\mathcal N=\sum_{s,t}\widetilde c_s^*\widetilde c_t
\langle d_s|d_t\rangle,
\]

with the exact coherent-state overlap

\[
\langle a|b\rangle=
\exp\!\left[-\frac{|a-b|^2}{2}
+i\bigl(\operatorname{Re}a\operatorname{Im}b
-\operatorname{Im}a\operatorname{Re}b\bigr)\right].
\]

The squared overlap with the larger complete shell of radius 20 is
`0.9999999999877`. This checks that radius 14 is a converged finite
representation of the chosen `Delta = 0.2` target. Its mean photon number,
computed from a 256-level projection, is about `12.2616`.

## Initial state, one circuit layer, and random starts

Every circuit starts from

\[
|\Psi_0\rangle=|0\rangle_q\otimes|0\rangle_o,
\]

meaning qubit ground state and oscillator vacuum. A depth `n` circuit has `n`
ECD-plus-qubit-rotation layers. One saved row is

```text
[beta_real, beta_imag, phi, theta, gamma = 0]
```

where the circuit displacement is `beta = beta_real + i beta_imag` and every
angle is in radians. In this repository's ECD convention, a layer first mixes
the two qubit branches and then displaces the oscillator by `-beta/2` on one
output branch and `+beta/2` on the other.

Each method receives the same three starting circuits at a fixed depth. For
every layer and restart,

\[
\beta\sim\mathcal{CN}(0,0.5),\qquad
\operatorname{Re}\beta,\operatorname{Im}\beta
\stackrel{\rm iid}{\sim}\mathcal N(0,0.25).
\]

Here `iid` means independent and identically distributed. This convention
means `E[|beta|^2] = 0.5`; it does not mean that each real
component has variance 0.5. The two optimized angles `phi` and `theta` are
independent uniform draws on `[0,2 pi]`. `gamma` is fixed to zero. Integer
seeds stored in `results.json` make these starts reproducible.

Adam is an adaptive gradient optimizer. Every restart receives 10,000 Adam
updates because improvements can arrive
well after 1,200 updates. These are best-observed results, not proofs of a
global optimum.

## Exact score used to rank every restart

After a circuit has been optimized, it is expanded exactly as

\[
|\Psi\rangle=\sum_{q=0}^1|q\rangle
\sum_j a_{qj}|z_{qj}\rangle .
\]

The qubit is not required to finish in a particular state. It is traced out,
so the common oscillator fidelity is

\[
F_{\rm exact}=\sum_{q=0}^1
\left|\sum_j a_{qj}\langle G|z_{qj}\rangle\right|^2.
\]

The target overlap `\<G|z>` is evaluated as the complete 323-site coherent
sum. It has no oscillator Fock cutoff. `1 - F_exact` is the exact infidelity.
Within each method and depth, the saved winner is the restart with largest
`F_exact`, never the largest internal surrogate score.

`Native fidelity` in `results.json` means the winning circuit's internal
48-level score evaluated by that method. It is useful for diagnosing cutoff or
compression bias, but it is not a headline performance metric.

An independent validation converts each winning physical circuit to 256 Fock
levels and compares it with the normalized 256-level target projection. The
validation discrepancy is reported below and in `results.json`.

## Shared 48-level training target

The Fock basis is the oscillator photon-number basis
`|0>, |1>, |2>, ...`. A 48-level Fock vector stores amplitudes only for photon
numbers 0 through 47.

Differentiating the full 323-site target sum through all `2^20` coherent paths
for 10,000 updates would make the full-path optimizer prohibitively slow. All
five methods therefore train against the same normalized projection of the
target onto Fock levels 0 through 47. This target projection retains

\[
\langle G|P_{48}|G\rangle=0.9673590488
\]

of the exact target norm. The full coherent circuit state itself is still
represented without a Fock cutoff; only its training target is projected.
The cutoff `N_f = 48` was chosen to match the packet methods' working grid
while keeping three 10,000-step dense-matrix restarts feasible.
This shared approximation makes the 10,000-step comparison practical, but it
is a real caveat: the optimizer does not directly reward the remaining 3.264%
of target norm. Exact 323-site rescoring prevents that approximation from
silently determining the reported winner.

## The five simulation and compression methods

### 1. Full coherent paths

For a coherent packet, displacement by `s` is analytic:

\[
c|z\rangle\longmapsto
c\,e^{-i\Omega(s,z)}|z+s\rangle,
\qquad
\Omega(a,b)=\operatorname{Re}a\operatorname{Im}b
-\operatorname{Im}a\operatorname{Re}b.
\]

The qubit rotation splits every existing path into two. Therefore depth `n`
has `2^n` path coefficients. No Fock state or displacement matrix is formed
during training. At depth 20 this means 1,048,576 analytic paths.

### 2. Analytic Fock recurrence

This method stores two arrays of 48 amplitudes, one for each qubit value. It
constructs the upper-left 48-by-48 block of the exact infinite-dimensional
displacement by

\[
D(a)|0\rangle=|a\rangle,
\qquad
D(a)|r\rangle=
\frac{(\hat a^\dagger-a^*)D(a)|r-1\rangle}{\sqrt r}.
\]

It then applies these blocks to the two Fock arrays. Because this is a
projection of an infinite unitary, the 48-by-48 block need not itself be
unitary: probability can leave the retained grid.

### 3. Dense numerical Fock

This is the requested materialized numerical benchmark. First truncate the
annihilation operator,

\[
a_{48}=\sum_{r=1}^{47}\sqrt r\,|r-1\rangle\langle r|.
\]

For every displacement, form the dense numerical unitary

\[
D^{\rm num}_{48}(a)=
\exp(a a_{48}^\dagger-a^*a_{48}).
\]

If `R_j` is the two-by-two qubit rotation in layer `j`, materialize the whole
96-by-96 joint layer matrix

\[
U_j=
\begin{pmatrix}
D^{\rm num}_{48}(-\beta_j/2)&0\\
0&D^{\rm num}_{48}(+\beta_j/2)
\end{pmatrix}
(R_j\otimes I_{48})
\]

and multiply a 96-entry joint state by `U_j`. This finite-grid displacement is
exactly unitary in 48 dimensions, but it has an artificial boundary because
the ladder operator was truncated before exponentiation. Matrix exponentials
and joint-state propagation use complex64; target-overlap arithmetic can
promote precision. Every saved circuit is rescored with the float64 cutoff-free
coherent formula.

The analytic and dense Fock methods therefore approximate the same physical
displacement in two different orders:

```text
analytic recurrence: infinite displacement -> take a 48-level block
dense numerical:     truncate ladder to 48 levels -> exponentiate
```

### 4. Continuous anchors

This method retains at most 32 coherent packets in each qubit branch, or 64
packet coefficients total. It also carries the two 48-entry Fock working
arrays used to define the projection and training score. Thus its propagated
state data contains 64 packet coefficients plus 96 Fock amplitudes, well below
the 1,024-coefficient memory ceiling. A layer can temporarily double the packet
count to 128 candidates before it is projected, so the largest such working
state has 128 packet coefficients plus 96 Fock amplitudes. After a layer
creates too many candidates:

```text
1. Convert the branch to a 48-level working vector v.
2. Greedily choose 32 candidate centers that reduce the residual.
3. Take 12 inner Adam steps that move those complex centers.
4. Refit every coefficient by a global regularized least-squares solve.
5. Carry only the 32 fitted packets into the next layer.
```

For basis matrix `B`, the coefficient refit is

\[
x=(B^\dagger B+10^{-9}I)^{-1}B^\dagger v,
\qquad |\widetilde v\rangle=Bx.
\]

### 5. Squeezed packets

This begins with the same movable coherent centers. It then takes 16 further
inner Adam steps to fit one bounded complex squeeze parameter per packet. The
state still retains 32 packets per qubit branch. This adds local shape freedom
but also adds an expensive inner optimization at every compression event.

The two packet methods use a straight-through outer gradient: their forward
pass uses the actually compressed state, while the derivative of the discrete
projection is approximated by the identity. This is a heuristic. It is useful
for optimization but is not an exact derivative and is never used as the
final score.

Compact pseudocode for the whole comparison is:

```text
for method in the five state representations:
    for depth in [4, 8, 12, 16, 20]:
        for restart in [0, 1, 2]:
            initialize the same random circuit for every method
            repeat 10,000 times:
                simulate with this method
                update circuit parameters to reduce 1 - F_48
            compute cutoff-free F_exact from the saved physical circuit
        save the restart with largest F_exact
```

## Results

Wall time is the total elapsed computer time for all three restarts at one
method and depth, including first JAX compilation and the cutoff-free host
rescore after each restart. It is not hardware gate time.

<!-- RESULTS_TABLE -->

<!-- RESULTS_DISCUSSION -->

## Gate-time definitions

The Coherax timing convention is

\[
T_C=\sum_j\left[24\,{\rm ns}+
\max\!\left(
\frac{|\beta_j|}{(2\pi\,50\,{\rm kHz})20},48\,{\rm ns}
\right)\right].
\]

The repository does not include the upstream Eickbusch pulse compiler for new
arbitrary sequences. The separate Eickbusch plot therefore uses the explicit
layerwise hardware lower bound

\[
T_{E,{\rm LB}}=\sum_j\max\!\left(
\frac{|\beta_j|}{(2\pi\,33\,{\rm kHz})30},224\,{\rm ns}
\right).
\]

The 224 ns floor is two 24 ns qubit pulses plus four 44 ns displacement
pulses. This is not a compiled waveform duration. Search wall time and
hardware gate time answer different questions and should not be compared as
if they had the same meaning.

## Wigner functions

[`figs/wigner_optimizer_grid.png`](figs/wigner_optimizer_grid.png) is generated
with `dq.plot.wigner`. Its five rows are the five optimizers, and its five
columns are the layer counts. The bottom panel is the normalized 96-level
projection of the exact finite target; that projection retains about
`0.9995326080` of the target norm. Every optimizer panel is reconstructed from
the saved physical circuit at 96 Fock levels and traced over the qubit. It is
not a picture of an internal compressed surrogate.

## Files

- [`hex_gkp_state_preparation_optimizers.ipynb`](hex_gkp_state_preparation_optimizers.ipynb):
  self-contained definitions, formulas, executable code, and artifact generation.
- [`results.json`](results.json): every restart score and time, selected
  winners, configuration, validation, and environment information.
- [`best_sequences.npz`](best_sequences.npz): all 25 winning parameter arrays
  in real and complex forms.
- [`sequences/`](sequences/): one readable JSON layer sequence per winner.
- [`figs/`](figs/): four requested line plots and the Wigner-function plate.
- [`report.pdf`](report.pdf): short self-contained report.
- [`workflow_hex_gkp_state_preparation_optimizers.sh`](workflow_hex_gkp_state_preparation_optimizers.sh):
  one-command reproduction under the repository memory and time guard.

## Reproduce

From this directory:

```bash
./workflow_hex_gkp_state_preparation_optimizers.sh
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
fresh processes on one allocation and writes atomic JSON/NPZ shards. The
workers run strictly sequentially, so at most one node and four CPU cores are
active. Launch using:

```bash
cluster/submit_cluster.sh
```

The dependent aggregation job selects each winner by exact analytic fidelity
and regenerates the experiment artifacts. See [`cluster/README.md`](cluster/README.md)
for the matched-block timing design, per-restart hardware records, post-hoc
hardware grouping, environment setup, monitoring, and safe resumption.
