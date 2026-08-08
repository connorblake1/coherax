"""Build the self-contained Experiment 06 notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
SOURCE = (HERE / "_optimization_experiment.py").read_text(encoding="utf-8")

SECTION_MARKDOWN = {
    "Explicit benchmark contract": r"""
## 1. Benchmark contract

The initial joint state is the qubit ground state and oscillator vacuum,

\[
|\Psi_0\rangle=|0\rangle_q\otimes|0\rangle_o.
\]

The oscillator target is the normalized adjacent-number superposition

\[
|\psi_{\rm target}\rangle=\frac{|9\rangle+|10\rangle}{\sqrt2}.
\]

Here \(|m\rangle\) means exactly \(m\) oscillator photons. The tested layer
counts are \(n\in\{4,8,12,16,20\}\). Every method receives the same three
deterministic starting sequences at a fixed depth. Depths 4--16 use 1,200 Adam
updates per restart. Depth 20 uses 10,000 updates because short Adam runs can
stop before a late optimization breakthrough. Therefore, method comparisons
at depth 20 are budget-matched, but the change from depth 16 to depth 20 changes
both circuit depth and optimization effort. It is not a depth-only comparison.
The two saved warm starts originally prepared \(|9\rangle\) and \(|10\rangle\),
but every update in this experiment uses the new superposition target.

The five method labels mean:

- **full coherent paths:** keep every analytic coherent-state path, so a depth
  \(n\) objective evaluation has \(2^n\) paths and no oscillator cutoff;
- **analytic Fock recurrence:** propagate two arrays of 48 Fock coefficients
  using exact infinite-dimensional displacement matrix elements;
- **dense numerical Fock:** truncate the annihilation operator to 48 levels,
  materialize every displacement with a matrix exponential, and propagate the
  full 96-entry qubit-oscillator state with materialized joint unitaries;
- **continuous anchors:** retain 32 movable coherent packets per qubit branch;
- **squeezed packets:** retain 32 movable squeezed packets per qubit branch.

The packet methods can temporarily have 64 candidates in each branch, then
project immediately back to 32. They never construct the full path tree.
""",
    "Circuit formulas and the exact coherent-path score": r"""
## 2. One circuit layer and the exact score

One saved row is \([\beta,\phi,\theta,\gamma]\), where \(\beta\) is complex,
the three other entries are angles in radians, and this benchmark fixes
\(\gamma=0\). A layer first applies the qubit rotation and then the echoed
conditional displacement (ECD). In the row-swapped ECD convention used by the
code, the two oscillator branches are displaced by \(-\beta/2\) and
\(+\beta/2\).

For a coherent packet \(c|z\rangle\), displacement by \(s\) gives

\[
c|z\rangle\longmapsto
c\exp[-i\,\operatorname{Im}(s^*z)]|z+s\rangle.
\]

After \(n\) layers the exact output is
\(|\Psi\rangle=\sum_{q=0}^1|q\rangle\sum_j c_{qj}|z_{qj}\rangle\).
Using

\[
\langle m|z\rangle=e^{-|z|^2/2}\frac{z^m}{\sqrt{m!}},
\]

the common exact score is

\[
F=\sum_{q=0}^1\left|
\sum_j c_{qj}e^{-|z_{qj}|^2/2}
\frac{z_{qj}^9/\sqrt{9!}+z_{qj}^{10}/\sqrt{10!}}{\sqrt2}
\right|^2.
\]

This is the fidelity of the reduced oscillator state with the pure target.
The qubit is traced out, which explains the sum over \(q\). Every restart,
including Fock and packet restarts, is ranked by this same cutoff-free score.
""",
    "Two Fock simulators": r"""
## 3. Two different Fock simulators

The Fock basis is the oscillator photon-number basis
\(\{|0\rangle,|1\rangle,\ldots\}\). A 48-level vector stores amplitudes only
for photon numbers 0 through 47.

Both methods store \(\psi_{q,m}=\langle q,m|\Psi\rangle\) for qubit
values \(q=0,1\) and oscillator levels \(m=0,\ldots,47\). The analytic method
constructs each projected displacement matrix from the recurrence

\[
D(a)|0\rangle=|a\rangle,\qquad
D(a)|k\rangle=\frac{(\hat a^\dagger-a^*)D(a)|k-1\rangle}{\sqrt{k}}.
\]

This **analytic Fock recurrence** returns the upper-left block of the exact
infinite-dimensional displacement. It is generally not unitary because
amplitude can leave the retained grid.

The new **dense numerical Fock** benchmark reverses the order of truncation and
exponentiation. It first forms

\[
a_{48}=\sum_{r=1}^{47}\sqrt r\,|r-1\rangle\langle r|,
\qquad
D_{48}^{\rm num}(a)=\exp(a a_{48}^\dagger-a^*a_{48}),
\]

then materializes the joint \(96\times96\) layer matrix

\[
\begin{pmatrix}D_{48}^{\rm num}(-\beta/2)&0\\
0&D_{48}^{\rm num}(+\beta/2)\end{pmatrix}(R\otimes I_{48}).
\]

This finite matrix is unitary but has a cutoff-dependent boundary. Dense matrix
exponentials and joint-state propagation use complex64; target-overlap arithmetic
can promote precision. Every saved sequence is rescored by the float64,
cutoff-free formula above. Both native objectives normalize their finite state.
A 96-level Fock conversion is also saved as an independent validation.
""",
    "Continuous-anchor and squeezed-packet objectives": r"""
## 4. Online packet objectives

Suppose a branch has candidate state \(|v\rangle\) and proposed packet columns
in a matrix \(B\). Both compressors globally refit the linear coefficients:

\[
a=(B^\dagger B+10^{-9}I)^{-1}B^\dagger v,
\qquad |\widetilde v\rangle=Ba.
\]

Candidate centers are seeded by greedy residual pivots. At each pivot, the
algorithm selects the candidate column with the largest squared overlap with
the current residual after removing the span of earlier pivots.

For **continuous anchors**, the 32 complex centers in each branch are then
moved by 12 inner Adam steps to maximize the normalized projection fidelity
between \(|v\rangle\) and \(|\widetilde v\rangle\).

For **squeezed packets**, the same center update is followed by 16 Adam steps
for one bounded complex squeeze \(\nu_j\) per packet. Its Fock coefficients are
generated by

\[
f_0=1,\qquad
f_{m+1}=\frac{(\mu z+\nu z^*)f_m-\nu\sqrt m\,f_{m-1}}
{\mu\sqrt{m+1}},\qquad \mu=\sqrt{1+|\nu|^2}.
\]

The forward objective uses the compressed state after every projection. The
outer circuit gradient uses a straight-through projection estimator: in the
backward pass the projection derivative is replaced by the identity. This
choice makes the discrete pivot selection trainable, but it is a heuristic;
the exact post-optimization score is therefore essential.

In compact pseudocode:

```
state = |qubit 0, oscillator 0>
packets = one coherent packet at z = 0
for layer in circuit:
    state = exact_Fock_step(state, layer)
    candidates = analytic_packet_step(packets, layer)
    if candidates_per_branch > 32:
        seeds = greedy_residual_pivots(candidates, state)
        centers = optimize_projection_centers(seeds, state)
        if squeezed: squeezes = optimize_squeezes(centers, state)
        packets, projected_state = global_least_squares_fit(state)
        state = projected_state
return normalized_target_overlap(state)
```
""",
    "Matched multirestart optimization": r"""
## 5. Multirestart selection and timers

For every method and depth, three complete optimizations are run. Each restart
is timed. A restart at depths 4--16 has 1,200 outer Adam updates, while a
restart at depth 20 has 10,000. The plotted wall time is the elapsed time for
all three restarts, including the first JAX compilation; it is the actual cost
paid to obtain the reported winner. The winner is the restart with the
smallest exact analytic infidelity \(1-F\), not the smallest internal surrogate
loss. The depth-20 random restart remains substantially worse than the warm
restart even after 10,000 steps, so the saved scores are seed-dependent best
observations rather than global-optimum claims.

The saved layer JSON files specify \(\operatorname{Re}\beta\),
\(\operatorname{Im}\beta\), \(\phi\), \(\theta\), and \(\gamma=0\) explicitly.
""",
    "Exact validation, plots, and saved artifacts": r"""
## 6. Timing formulas, Wigner functions, and outputs

The same winning physical sequence is assigned two hardware-time estimates.
The Coherax convention is

\[
T_C=\sum_i\left[24\,{\rm ns}+\max\!\left(
\frac{|\beta_i|}{(2\pi\,50\,{\rm kHz})20},48\,{\rm ns}\right)\right].
\]

The repository does not contain the upstream Eickbusch pulse compiler for new
arbitrary sequences. We therefore report a plainly labeled layerwise lower
bound under their hardware numbers,

\[
T_{E,{\rm LB}}=\sum_i\max\!\left(
\frac{|\beta_i|}{(2\pi\,33\,{\rm kHz})30},224\,{\rm ns}\right).
\]

It is not a compiled waveform duration. The 224 ns floor equals two 24 ns
qubit pulses plus four 44 ns displacement pulses.

Dynamiqs `dq.plot.wigner` draws the exact reduced oscillator density matrix of
every saved winning circuit. The 5-by-5 optimizer/depth array is followed by a
single bottom panel containing the exact target. The packet surrogate itself is
not plotted; each image is reconstructed from the saved physical circuit by
the full coherent-path formula.
""",
}

chunks = SOURCE.split("# %% ")
cells = [
    nbf.v4.new_markdown_cell(
        r"""# Experiment 06: end-to-end state preparation by five optimizers

This notebook asks five state representations to optimize the same ECD circuit
for the same target, \((|9\rangle+|10\rangle)/\sqrt2\), at depths 4, 8, 12,
16, and 20. It contains every experiment-specific formula, optimizer, validation,
timer, saved sequence, plot, and Wigner-function call needed to reproduce the
report; no source-code reading is required to understand the labels."""
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
nbf.write(notebook, HERE / "state_preparation_optimizers.ipynb")
print("wrote state_preparation_optimizers.ipynb")
