"""Build the self-contained Experiment 07 notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
SOURCE = (HERE / "_optimization_experiment.py").read_text(encoding="utf-8")

SECTION_MARKDOWN = {
    "Explicit benchmark contract": r"""
## 1. Exact benchmark contract

The initial joint state is qubit ground state times oscillator vacuum,

\[
|\Psi_0\rangle=|0\rangle_q\otimes|0\rangle_o.
\]

The target is the logical-zero finite-energy hexagonal
Gottesman--Kitaev--Preskill (GKP) state with envelope parameter
\(\Delta=0.2\). Define the two hexagonal lattice vectors

\[
\alpha=\sqrt{\pi/\sqrt3},\qquad
\beta=e^{2\pi i/3}\alpha.
\]

The numerical target retains the complete hexagonal shell of radius \(R=14\).
For every integer pair \((m,l)\) satisfying

\[
\max(|m|,|l|,|m-l|)\leq14,\qquad m\ \text{even},
\]

write \(k=m/2\), \(d_{ml}=m\alpha+l\beta\), and

\[
\widetilde c_{ml}=\exp[-i\pi kl-\Delta^2|d_{ml}|^2].
\]

The normalized target is

\[
|G_{0,\Delta}^{(14)}\rangle=
\mathcal N^{-1/2}\sum_{m,l}\widetilde c_{ml}|d_{ml}\rangle,
\]

where \(\mathcal N\) is computed with the exact coherent-state Gram matrix.
This shell contains 323 coherent sites. Its exact coherent overlap with the
larger \(R=20\) shell has squared magnitude 0.9999999999877, so shell error is
negligible on the scale of this optimizer comparison.

The tested circuit depths are \(n\in\{4,8,12,16,20\}\). Every method receives
the same three initial circuits at a fixed depth and performs 10,000 outer Adam
updates per restart. Adam is an adaptive gradient optimizer.

For every layer and restart, the controlled displacement is sampled as

\[
\beta_j\sim\mathcal{CN}(0,0.5),\qquad
\operatorname{Re}\beta_j,\operatorname{Im}\beta_j
\stackrel{\mathrm{iid}}{\sim}\mathcal N(0,0.25).
\]

Thus \(\mathbb E|\beta_j|^2=0.5\). The angles \(\phi_j\) and \(\theta_j\) are
independent uniform draws on \([0,2\pi]\). The experiment fixes
\(\gamma_j=0\), leaving four optimized real numbers per layer. Fixed integer seeds make every draw
reproducible. Here ``iid'' means independent and identically distributed.
""",
    "Circuit formulas and the exact coherent-path score": r"""
## 2. One circuit layer, training score, and exact score

One saved row is \([\beta,\phi,\theta,\gamma]\), where \(\beta\) is complex,
the three other entries are angles in radians, and \(\gamma=0\). The layer
applies a qubit rotation followed by an echoed conditional displacement. In
this repository's row-swapped ECD convention, the two oscillator branches are
displaced by \(-\beta/2\) and \(+\beta/2\).

For coherent packets,

\[
c|z\rangle\mapsto
c\exp[-i\,\operatorname{Im}(s^*z)]|z+s\rangle.
\]

After \(n\) layers the exact circuit output is

\[
|\Psi\rangle=\sum_{q=0}^1|q\rangle\sum_j a_{qj}|z_{qj}\rangle.
\]

The common final score traces out the qubit and uses the complete 323-site
target:

\[
F_{\rm exact}=\sum_{q=0}^1\left|
\sum_j a_{qj}\langle G_{0,0.2}^{(14)}|z_{qj}\rangle
\right|^2,
\]

\[
\langle G|z\rangle=
\sum_t c_t^*\exp[-|d_t-z|^2/2+i\,\Omega(d_t,z)],
\quad
\Omega(a,b)=\operatorname{Re}a\operatorname{Im}b-
\operatorname{Im}a\operatorname{Re}b.
\]

Every saved winner is selected by this cutoff-free oscillator score.

The full exact target sum is too expensive to differentiate through 10,000
times at depth 20. Therefore every method trains against the same normalized
48-level target projection. Its retained target norm is
\(\langle G|P_{48}|G\rangle=0.9673590488\). This is a shared training
approximation, not the reported score. The exact 323-site score above is
computed after every restart and determines the winner.
""",
    "Two Fock simulators": r"""
## 3. Two different Fock simulators

The Fock basis is the oscillator photon-number basis
\(\{|0\rangle,|1\rangle,\ldots\}\). A 48-level vector stores amplitudes only
for photon numbers 0 through 47.

Both Fock optimizers use \(N_f=48\), but they implement displacement in
different orders. This distinction is the new benchmark requested here.

**Analytic Fock recurrence.** It evaluates the exact infinite-dimensional
matrix elements and keeps their upper-left \(48\times48\) block:

\[
D(a)|0\rangle=|a\rangle,\qquad
D(a)|k\rangle=
\frac{(\hat a^\dagger-a^*)D(a)|k-1\rangle}{\sqrt{k}}.
\]

This projected matrix is generally not unitary because amplitude can leave the
48-level grid.

**Dense numerical Fock.** It first truncates the annihilation operator,

\[
a_{48}=\sum_{r=1}^{47}\sqrt r\,|r-1\rangle\langle r|,
\]

then materializes

\[
D_{48}^{\rm num}(a)=
\exp(a a_{48}^\dagger-a^*a_{48}).
\]

It constructs the full joint layer matrix

\[
U_j=
\begin{pmatrix}D_{48}^{\rm num}(-\beta_j/2)&0\\
0&D_{48}^{\rm num}(+\beta_j/2)\end{pmatrix}
(R_j\otimes I_{48})
\]

and multiplies the 96-entry joint state by \(U_j\). This finite matrix is
unitary, but its boundary behavior differs from the projected infinite
displacement. Matrix exponentials and state propagation use complex64;
target-overlap arithmetic can promote precision. The saved physical circuit
is rescored in float64 by the exact coherent formula.
""",
    "Continuous-anchor and squeezed-packet objectives": r"""
## 4. Full paths and the two online compressors

**Full coherent paths** retains every analytic path. A depth-\(n\) state has
\(2^n\) paths and no oscillator cutoff. Its training overlap with the
48-level target is evaluated analytically from
\(\langle r|z\rangle=e^{-|z|^2/2}z^r/\sqrt{r!}\).

**Continuous anchors** and **squeezed packets** retain 32 packets in each
qubit branch, or 64 packet coefficients total. If a branch has more than 32
candidates, greedy residual pivots select seeds and a global least-squares fit
computes the coefficients,

\[
a=(B^\dagger B+10^{-9}I)^{-1}B^\dagger v,
\qquad |\widetilde v\rangle=Ba.
\]

Continuous anchors take 12 inner Adam steps to move the complex packet centers.
Squeezed packets add 16 steps for one bounded complex squeeze per packet. Their
outer circuit gradient uses a straight-through projection estimator: the
backward derivative of the discrete projection is replaced by the identity.
This is heuristic, so the internal score never selects a winner.

Compact pseudocode:

```
for method, depth, restart:
    parameters = requested_random_initialization(depth, restart)
    repeat 10,000 Adam steps:
        propagate state with this method
        loss = 1 - fidelity_with_normalized_48_level_target
    exact_F = overlap_with_323_site_target(parameters)
select restart with largest exact_F
```
""",
    "Matched multirestart optimization": r"""
## 5. Matched restarts, wall time, and saved circuits

At each depth, the five methods receive exactly the same three random initial
parameter arrays. Each restart receives 10,000 Adam updates. The plotted wall
time is the measured elapsed time for all three restarts, including the first
JAX compilation and the cutoff-free host rescore after each restart. It is the
actual search cost paid to obtain the displayed winner.

Every layer JSON explicitly saves \(\operatorname{Re}\beta\),
\(\operatorname{Im}\beta\), \(\phi\), \(\theta\), and \(\gamma=0\). Native
training fidelity means the method's own normalized 48-level score. It and the
exact final fidelity are both retained in `results.json` so cutoff and
compression bias remain visible.
""",
    "Exact validation, plots, and saved artifacts": r"""
## 6. Gate-time formulas, validation, and Wigner plots

The Coherax hardware-time estimate is

\[
T_C=\sum_i\left[24\,{\rm ns}+\max\!\left(
\frac{|\beta_i|}{(2\pi\,50\,{\rm kHz})20},48\,{\rm ns}\right)\right].
\]

The separate Eickbusch-hardware plot is explicitly a lower bound,

\[
T_{E,{\rm LB}}=\sum_i\max\!\left(
\frac{|\beta_i|}{(2\pi\,33\,{\rm kHz})30},224\,{\rm ns}\right).
\]

It is not an upstream compiled waveform duration. Optimization wall time and
hardware gate time are different quantities.

Every saved circuit is independently converted to 256 Fock levels and compared
with the normalized 256-level target projection. Wigner functions use 96 Fock
levels; that target projection retains 0.9995326080 of the exact target norm.
The Wigner plate contains five optimizer rows, five depth columns, and one
reference target panel at the bottom. Every optimizer panel is reconstructed
from the saved physical circuit, not from a compressed surrogate.
""",
}

chunks = SOURCE.split("# %% ")
cells = [
    nbf.v4.new_markdown_cell(
        r"""# Experiment 07: finite hex-GKP state preparation

This notebook compares five state representations while optimizing the same
ECD circuit for the same logical-zero hexagonal GKP target with
\(\Delta=0.2\). It defines the target, initialization, layer convention,
training approximation, exact ranking score, both Fock implementations,
compression routines, timers, validations, plots, and saved files. No source
code outside this notebook is needed to understand an optimizer label or a
headline number."""
    ),
    nbf.v4.new_code_cell(chunks[0].rstrip()),
]

for chunk in chunks[1:]:
    title, body = chunk.split("\n", 1)
    title = title.strip()
    cells.append(nbf.v4.new_markdown_cell(SECTION_MARKDOWN.get(title, f"## {title}")))
    cells.append(nbf.v4.new_code_cell(f"# %% {title}\n{body.rstrip()}"))

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "coherax",
            "language": "python",
            "name": "coherax",
        },
        "language_info": {"name": "python", "version": "3"},
    },
)
nbf.write(notebook, HERE / "hex_gkp_state_preparation_optimizers.ipynb")
print("wrote hex_gkp_state_preparation_optimizers.ipynb")
