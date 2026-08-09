"""Build the self-contained Experiment 09 aggregation notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
README = (HERE / "README.md").read_text(encoding="utf-8")
SOURCE = (HERE / "_aggregate_results.py").read_text(encoding="utf-8")

SECTION_INTRODUCTIONS = {
    "Explicit aggregation contract": r"""
## Executable contract

The following constants make the expected targets, methods, depths, restart
count, fidelity tolerances, plotting floor, colors, and source paths explicit.
The plotting floor affects only logarithmic display; it never changes a stored
fidelity or winner.
""",
    "Strict source validation and flattening": r"""
## Input validation

For every source target, validation requires exactly 25 unique method/depth
records and exactly 100 uniquely indexed restarts inside each record. It checks
that `exact_infidelity` equals `1 - exact_fidelity`, that the saved winner is
the maximum exact-fidelity restart, and that the saved total time equals the sum
of restart times. Every source file is hashed with SHA-256 so this report names
the exact input bytes it analyzed.
""",
    "Statistical definitions": r"""
## Statistics

The next functions calculate medians, interquartile ranges, 95 percent Wilson
intervals for restart success, fractional win shares for exact ties, and paired
timing ratios. Timing ratios use only restart blocks whose source hardware audit
says that all 25 cases ran under one hardware signature.
""",
    "Cross-target plots": r"""
## Plots

Each target receives its own panel. This prevents a raw wall-time curve from
suggesting that runs on different CPU models are directly controlled. The
paired-speedup plot is the hardware-matched timing comparison.
""",
    "Tables and machine-readable artifacts": r"""
## Saved artifacts

The aggregation writes all 7,500 restart rows, the 75 best-of-100 summaries,
five cross-task method summaries, exact source-file hashes, a source-artifact
inventory, report tables, and the figures described above.
""",
    "Reproduction entry point": r"""
## Execute the aggregation

This final cell uses the repository containing the notebook and writes outputs
into this Experiment 09 directory. It performs no optimization and does not
modify Experiments 06--08.
""",
}


chunks = SOURCE.split("# %% ")
cells: list[nbf.NotebookNode] = [
    nbf.v4.new_markdown_cell(README),
    nbf.v4.new_code_cell(
        "_EXPERIMENT_09_NOTEBOOK = True\n\n" + chunks[0].rstrip()
    ),
]
for chunk in chunks[1:]:
    title, body = chunk.split("\n", 1)
    title = title.strip()
    cells.append(
        nbf.v4.new_markdown_cell(
            SECTION_INTRODUCTIONS.get(title, f"## {title}")
        )
    )
    cells.append(nbf.v4.new_code_cell(f"# %% {title}\n{body.rstrip()}"))

cells.append(
    nbf.v4.new_code_cell(
        """notebook_result = run_aggregation(
    repository_root=Path.cwd().parents[1],
    output_directory=Path.cwd(),
    expected_restart_count=EXPECTED_RESTART_COUNT,
    create_plots=True,
)
print(
    "Validated and aggregated",
    notebook_result["input_contract"]["expected_individual_restart_records"],
    "restart records.",
)"""
    )
)

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "coherax",
            "language": "python",
            "name": "coherax",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
)
nbf.write(notebook, HERE / "cross_target_optimizer_aggregation.ipynb")
print("Wrote cross_target_optimizer_aggregation.ipynb")
