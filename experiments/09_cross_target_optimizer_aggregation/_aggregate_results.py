"""Aggregate the 100-restart cluster results from Experiments 06--08.

This file is copied into the self-contained Experiment 09 notebook by
``_build_notebook.py``. The notebook is the canonical executable record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# %% Explicit aggregation contract

EXPECTED_RESTART_COUNT = 100
EXPECTED_LAYERS = (4, 8, 12, 16, 20)
METHOD_NAMES = (
    "full_coherent",
    "direct_fock",
    "dense_numerical_fock",
    "squeezed_packets",
    "continuous_anchors",
)
METHOD_LABELS = {
    "full_coherent": "full coherent paths",
    "direct_fock": "analytic Fock recurrence",
    "dense_numerical_fock": "dense numerical Fock",
    "squeezed_packets": "squeezed packets",
    "continuous_anchors": "continuous anchors",
}
METHOD_SHORT_LABELS = {
    "full_coherent": "full coherent",
    "direct_fock": "analytic Fock",
    "dense_numerical_fock": "dense Fock",
    "squeezed_packets": "squeezed",
    "continuous_anchors": "anchors",
}
METHOD_COLORS = {
    "full_coherent": "#2667a9",
    "direct_fock": "#d1495b",
    "dense_numerical_fock": "#7b2cbf",
    "squeezed_packets": "#2a9d8f",
    "continuous_anchors": "#e09f3e",
}
METHOD_MARKERS = {
    "full_coherent": "o",
    "direct_fock": "s",
    "dense_numerical_fock": "^",
    "squeezed_packets": "D",
    "continuous_anchors": "P",
}
SUCCESS_THRESHOLDS = (0.9, 0.99, 0.999)
PLOTTING_INFIDELITY_FLOOR = 1e-16
FIDELITY_TOLERANCE = 1e-9
TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class TargetSpec:
    """Describe one source experiment and its state-preparation target."""

    key: str
    label: str
    short_label: str
    source_directory: str
    target_formula: str

    @property
    def aggregate_relative_path(self) -> Path:
        """Return the source aggregate path relative to the repository root."""
        return Path("experiments") / self.source_directory / "cluster" / "aggregate_results.json"


TARGET_SPECS = (
    TargetSpec(
        key="fock_superposition",
        label=r"Fock superposition $(|9\rangle+|10\rangle)/\sqrt{2}$",
        short_label=r"$(|9\rangle+|10\rangle)/\sqrt{2}$",
        source_directory="06_state_preparation_optimizers",
        target_formula="(|9> + |10>) / sqrt(2)",
    ),
    TargetSpec(
        key="hex_gkp_delta_0p2",
        label=r"finite hex-GKP logical zero, $\Delta=0.2$",
        short_label=r"hex GKP, $\Delta=0.2$",
        source_directory="07_hex_gkp_state_preparation_optimizers",
        target_formula="finite hex-GKP logical |0_L>, Delta=0.2, shell radius 14",
    ),
    TargetSpec(
        key="even_cat_alpha_2",
        label=r"even cat $(|-2\rangle+|2\rangle)/\sqrt{2+2e^{-8}}$",
        short_label=r"even cat, $\alpha=2$",
        source_directory="08_cat_state_preparation_optimizers",
        target_formula="(|-2> + |2>) / sqrt(2 + 2*exp(-8))",
    ),
)


def repository_root_from_script() -> Path:
    """Return the repository root inferred from this source file."""
    return Path(__file__).resolve().parents[2]


def file_sha256(path: Path) -> str:
    """Return a hexadecimal SHA-256 digest for one input file."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value: Any, field: str) -> float:
    """Convert a value to a finite float or raise a precise error."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {result}")
    return result


def restart_wall_seconds(restart: dict[str, Any]) -> float:
    """Return the full optimizer-plus-rescore duration for one restart."""
    if "total_wall_seconds" in restart:
        return finite_float(restart["total_wall_seconds"], "total_wall_seconds")
    if "wall_seconds" in restart:
        return finite_float(restart["wall_seconds"], "wall_seconds")
    raise KeyError("restart record has neither total_wall_seconds nor wall_seconds")


def exact_infidelity(fidelity: float) -> float:
    """Return ``1 - fidelity`` with only presentation-level clipping."""
    return max(0.0, 1.0 - fidelity)


# %% Strict source validation and flattening

def load_and_validate_source(
    repository_root: Path,
    target: TargetSpec,
    expected_restart_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and strictly validate one cluster aggregate.

    The validation prevents a partially copied cluster run, a local three-
    restart result, or a duplicated method/depth record from entering the
    cross-target comparison.
    """
    input_path = repository_root / target.aggregate_relative_path
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing Experiment {target.source_directory[:2]} cluster aggregate: "
            f"{input_path}\nRun its cluster aggregation with --publish, then copy "
            "the resulting cluster directory back into this checkout."
        )
    with input_path.open(encoding="utf-8") as input_file:
        aggregate = json.load(input_file)

    source_restart_count = int(aggregate.get("restart_count", -1))
    if source_restart_count != expected_restart_count:
        raise ValueError(
            f"{input_path} has restart_count={source_restart_count}; expected "
            f"exactly {expected_restart_count}"
        )
    records = aggregate.get("records")
    if not isinstance(records, list):
        raise TypeError(f"{input_path}: records must be a list")
    expected_pairs = {
        (method_name, layer_count)
        for method_name in METHOD_NAMES
        for layer_count in EXPECTED_LAYERS
    }
    if len(records) != len(expected_pairs):
        raise ValueError(
            f"{input_path} has {len(records)} winner records; expected "
            f"{len(expected_pairs)}"
        )

    seen_pairs: set[tuple[str, int]] = set()
    for record in records:
        method_name = str(record.get("method"))
        layer_count = int(record.get("layers", -1))
        pair = (method_name, layer_count)
        if pair not in expected_pairs:
            raise ValueError(f"{input_path}: unexpected method/depth pair {pair}")
        if pair in seen_pairs:
            raise ValueError(f"{input_path}: duplicate method/depth pair {pair}")
        seen_pairs.add(pair)
        if int(record.get("restart_count", -1)) != expected_restart_count:
            raise ValueError(f"{input_path}: {pair} has an incorrect restart_count")
        restarts = record.get("restarts")
        if not isinstance(restarts, list) or len(restarts) != expected_restart_count:
            raise ValueError(
                f"{input_path}: {pair} has {len(restarts) if isinstance(restarts, list) else 'invalid'} "
                f"restart records; expected {expected_restart_count}"
            )
        restart_indices = {int(item.get("restart", -1)) for item in restarts}
        if restart_indices != set(range(expected_restart_count)):
            raise ValueError(f"{input_path}: {pair} has missing or duplicate restart indices")

        restart_fidelities = []
        restart_times = []
        for restart in restarts:
            fidelity = finite_float(restart["exact_fidelity"], "restart exact_fidelity")
            if not -FIDELITY_TOLERANCE <= fidelity <= 1.0 + FIDELITY_TOLERANCE:
                raise ValueError(f"{input_path}: unphysical exact fidelity {fidelity}")
            stored_infidelity = finite_float(
                restart["exact_infidelity"], "restart exact_infidelity"
            )
            if not math.isclose(
                stored_infidelity,
                1.0 - fidelity,
                rel_tol=1e-8,
                abs_tol=1e-10,
            ):
                raise ValueError(f"{input_path}: inconsistent exact infidelity in {pair}")
            restart_fidelities.append(fidelity)
            duration = restart_wall_seconds(restart)
            if duration < 0.0:
                raise ValueError(f"{input_path}: negative restart wall time in {pair}")
            restart_times.append(duration)

        winner_fidelity = finite_float(record["exact_fidelity"], "winner exact_fidelity")
        if not math.isclose(
            winner_fidelity,
            max(restart_fidelities),
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{input_path}: {pair} is not the maximum-fidelity restart")
        stored_total = finite_float(
            record["total_multirestart_wall_seconds"],
            "total_multirestart_wall_seconds",
        )
        if not math.isclose(stored_total, sum(restart_times), rel_tol=1e-8, abs_tol=1e-6):
            raise ValueError(f"{input_path}: {pair} has an inconsistent total wall time")

    if seen_pairs != expected_pairs:
        missing = sorted(expected_pairs - seen_pairs)
        raise ValueError(f"{input_path}: missing method/depth pairs: {missing}")

    fairness = aggregate.get("timing_fairness", {})
    hardware_by_restart = fairness.get("hardware_by_restart", {})
    fair_restart_indices = []
    for restart_index in range(expected_restart_count):
        hardware = hardware_by_restart.get(str(restart_index), {})
        if hardware.get("matched_single_hardware") is True:
            fair_restart_indices.append(restart_index)

    manifest = {
        "target": target.key,
        "path": str(input_path.relative_to(repository_root)),
        "sha256": file_sha256(input_path),
        "restart_count": source_restart_count,
        "winner_record_count": len(records),
        "individual_restart_record_count": len(records) * source_restart_count,
        "fair_timing_restart_count": len(fair_restart_indices),
        "within_restart_hardware_matched": fairness.get(
            "within_restart_hardware_matched"
        ),
        "hardware_group_count": fairness.get("hardware_group_count"),
    }
    return aggregate, manifest


def flatten_restarts(
    target: TargetSpec,
    aggregate: dict[str, Any],
) -> list[dict[str, Any]]:
    """Flatten nested restart records and retain timing/hardware provenance."""
    rows: list[dict[str, Any]] = []
    for winner_record in aggregate["records"]:
        for restart in winner_record["restarts"]:
            hardware = restart.get("hardware", {})
            rows.append(
                {
                    "target": target.key,
                    "target_label": target.target_formula,
                    "method": winner_record["method"],
                    "method_label": winner_record.get(
                        "method_label", METHOD_LABELS[winner_record["method"]]
                    ),
                    "layers": int(winner_record["layers"]),
                    "restart": int(restart["restart"]),
                    "outer_steps": int(restart.get("outer_steps", 0)),
                    "native_fidelity": finite_float(
                        restart["native_fidelity"], "native_fidelity"
                    ),
                    "exact_fidelity": finite_float(
                        restart["exact_fidelity"], "exact_fidelity"
                    ),
                    "exact_infidelity": finite_float(
                        restart["exact_infidelity"], "exact_infidelity"
                    ),
                    "optimizer_wall_seconds": finite_float(
                        restart.get("optimizer_wall_seconds", restart_wall_seconds(restart)),
                        "optimizer_wall_seconds",
                    ),
                    "exact_rescore_wall_seconds": finite_float(
                        restart.get("exact_rescore_wall_seconds", 0.0),
                        "exact_rescore_wall_seconds",
                    ),
                    "total_wall_seconds": restart_wall_seconds(restart),
                    "node": hardware.get("node"),
                    "cpu_model": hardware.get("cpu_model"),
                    "partition": hardware.get("partition"),
                    "cpus_per_task": hardware.get("cpus_per_task"),
                    "python_version": hardware.get("python_version"),
                    "numpy_version": hardware.get("numpy_version"),
                    "jax_version": hardware.get("jax_version"),
                    "dynamiqs_version": hardware.get("dynamiqs_version"),
                    "conda_environment": hardware.get("conda_environment"),
                }
            )
    return rows


# %% Statistical definitions

def quantile(values: Iterable[float], probability: float) -> float:
    """Return a float quantile from a finite nonempty collection."""
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        return math.nan
    return float(np.quantile(array, probability))


def wilson_interval(success_count: int, sample_count: int) -> tuple[float, float]:
    """Return the two-sided 95 percent Wilson binomial interval."""
    if sample_count <= 0:
        return math.nan, math.nan
    z_value = 1.959963984540054
    probability = success_count / sample_count
    denominator = 1.0 + z_value**2 / sample_count
    center = (probability + z_value**2 / (2.0 * sample_count)) / denominator
    half_width = (
        z_value
        * math.sqrt(
            probability * (1.0 - probability) / sample_count
            + z_value**2 / (4.0 * sample_count**2)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def fair_restart_indices(
    aggregate: dict[str, Any], expected_restart_count: int
) -> list[int]:
    """Return restart blocks whose 25 cases used one hardware signature."""
    hardware_by_restart = aggregate.get("timing_fairness", {}).get(
        "hardware_by_restart", {}
    )
    return [
        restart_index
        for restart_index in range(expected_restart_count)
        if hardware_by_restart.get(str(restart_index), {}).get(
            "matched_single_hardware"
        )
        is True
    ]


def record_lookup(aggregate: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    """Index winner records by method and layer count."""
    return {
        (record["method"], int(record["layers"])): record
        for record in aggregate["records"]
    }


def restart_lookup(record: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Index one method/depth record by restart number."""
    return {int(item["restart"]): item for item in record["restarts"]}


def summarize_winners(
    sources: dict[str, dict[str, Any]],
    expected_restart_count: int,
) -> list[dict[str, Any]]:
    """Compute winner quality, restart robustness, and fair timing ratios."""
    summaries: list[dict[str, Any]] = []
    target_by_key = {target.key: target for target in TARGET_SPECS}
    for target_key, aggregate in sources.items():
        target = target_by_key[target_key]
        indexed = record_lookup(aggregate)
        eligible_timing = fair_restart_indices(aggregate, expected_restart_count)
        for layer_count in EXPECTED_LAYERS:
            baseline_restarts = restart_lookup(indexed[("full_coherent", layer_count)])
            restart_fidelity_by_method = {
                method_name: {
                    int(item["restart"]): finite_float(
                        item["exact_fidelity"], "exact_fidelity"
                    )
                    for item in indexed[(method_name, layer_count)]["restarts"]
                }
                for method_name in METHOD_NAMES
            }
            win_shares = {method_name: 0.0 for method_name in METHOD_NAMES}
            for restart_index in range(expected_restart_count):
                scores = {
                    method_name: restart_fidelity_by_method[method_name][restart_index]
                    for method_name in METHOD_NAMES
                }
                maximum = max(scores.values())
                tied = [
                    method_name
                    for method_name, score in scores.items()
                    if abs(score - maximum) <= TIE_TOLERANCE
                ]
                for method_name in tied:
                    win_shares[method_name] += 1.0 / len(tied)

            best_scores = {
                method_name: finite_float(
                    indexed[(method_name, layer_count)]["exact_fidelity"],
                    "exact_fidelity",
                )
                for method_name in METHOD_NAMES
            }
            for method_name in METHOD_NAMES:
                record = indexed[(method_name, layer_count)]
                restarts = restart_lookup(record)
                restart_fidelities = np.asarray(
                    [
                        finite_float(restarts[index]["exact_fidelity"], "exact_fidelity")
                        for index in range(expected_restart_count)
                    ],
                    dtype=np.float64,
                )
                restart_infidelities = np.maximum(0.0, 1.0 - restart_fidelities)
                restart_times = np.asarray(
                    [restart_wall_seconds(restarts[index]) for index in range(expected_restart_count)],
                    dtype=np.float64,
                )
                speedups = np.asarray(
                    [
                        restart_wall_seconds(baseline_restarts[index])
                        / max(restart_wall_seconds(restarts[index]), 1e-15)
                        for index in eligible_timing
                    ],
                    dtype=np.float64,
                )
                best_fidelity = finite_float(record["exact_fidelity"], "exact_fidelity")
                best_infidelity = exact_infidelity(best_fidelity)
                summary: dict[str, Any] = {
                    "target": target.key,
                    "target_label": target.target_formula,
                    "method": method_name,
                    "method_label": METHOD_LABELS[method_name],
                    "layers": layer_count,
                    "restart_count": expected_restart_count,
                    "winning_restart": int(record["winning_restart"]),
                    "best_exact_fidelity": best_fidelity,
                    "best_exact_infidelity": best_infidelity,
                    "fidelity_rank": 1
                    + sum(
                        score > best_fidelity + TIE_TOLERANCE
                        for score in best_scores.values()
                    ),
                    "restart_exact_fidelity_median": float(
                        np.median(restart_fidelities)
                    ),
                    "restart_exact_fidelity_q25": quantile(restart_fidelities, 0.25),
                    "restart_exact_fidelity_q75": quantile(restart_fidelities, 0.75),
                    "restart_exact_infidelity_median": float(
                        np.median(restart_infidelities)
                    ),
                    "restart_win_share": win_shares[method_name]
                    / expected_restart_count,
                    "total_multirestart_wall_seconds": float(np.sum(restart_times)),
                    "restart_wall_seconds_median": float(np.median(restart_times)),
                    "restart_wall_seconds_q25": quantile(restart_times, 0.25),
                    "restart_wall_seconds_q75": quantile(restart_times, 0.75),
                    "fair_timing_restart_count": int(speedups.size),
                    "paired_speedup_vs_full_coherent_median": (
                        float(np.median(speedups)) if speedups.size else math.nan
                    ),
                    "paired_speedup_vs_full_coherent_q25": quantile(speedups, 0.25),
                    "paired_speedup_vs_full_coherent_q75": quantile(speedups, 0.75),
                    "coherax_gate_time_microseconds": finite_float(
                        record["coherax_gate_time_microseconds"],
                        "coherax_gate_time_microseconds",
                    ),
                    "eickbusch_lower_bound_microseconds": finite_float(
                        record["eickbusch_lower_bound_microseconds"],
                        "eickbusch_lower_bound_microseconds",
                    ),
                }
                for threshold in SUCCESS_THRESHOLDS:
                    suffix = str(threshold).replace(".", "p")
                    success_count = int(np.count_nonzero(restart_fidelities >= threshold))
                    lower, upper = wilson_interval(success_count, expected_restart_count)
                    summary[f"success_count_f_ge_{suffix}"] = success_count
                    summary[f"success_rate_f_ge_{suffix}"] = (
                        success_count / expected_restart_count
                    )
                    summary[f"success_rate_f_ge_{suffix}_wilson_low"] = lower
                    summary[f"success_rate_f_ge_{suffix}_wilson_high"] = upper
                summaries.append(summary)
    return summaries


def summarize_methods(winner_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize each representation across all 15 target/depth tasks."""
    method_rows: list[dict[str, Any]] = []
    for method_name in METHOD_NAMES:
        records = [row for row in winner_summaries if row["method"] == method_name]
        infidelities = np.asarray(
            [max(row["best_exact_infidelity"], PLOTTING_INFIDELITY_FLOOR) for row in records],
            dtype=np.float64,
        )
        speedups = np.asarray(
            [
                row["paired_speedup_vs_full_coherent_median"]
                for row in records
                if math.isfinite(row["paired_speedup_vs_full_coherent_median"])
            ],
            dtype=np.float64,
        )
        method_rows.append(
            {
                "method": method_name,
                "method_label": METHOD_LABELS[method_name],
                "task_count": len(records),
                "mean_fidelity_rank": float(
                    np.mean([row["fidelity_rank"] for row in records])
                ),
                "best_of_five_task_count": int(
                    sum(row["fidelity_rank"] == 1 for row in records)
                ),
                "geometric_mean_best_infidelity": float(
                    np.exp(np.mean(np.log(infidelities)))
                ),
                "median_fair_speedup_vs_full_coherent": (
                    float(np.median(speedups)) if speedups.size else math.nan
                ),
                "total_optimizer_process_hours": float(
                    sum(row["total_multirestart_wall_seconds"] for row in records)
                    / 3600.0
                ),
            }
        )
    return method_rows


# %% Cross-target plots

def configure_plots() -> None:
    """Set compact, readable plotting defaults."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "figure.dpi": 130,
            "savefig.bbox": "tight",
        }
    )


def rows_for_target(
    summaries: list[dict[str, Any]], target_key: str
) -> list[dict[str, Any]]:
    """Return winner-summary rows for one target."""
    return [row for row in summaries if row["target"] == target_key]


def add_shared_legend(figure: plt.Figure, axes: np.ndarray) -> None:
    """Add one method legend below a multi-panel figure."""
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(METHOD_NAMES),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )


def plot_lines_by_target(
    summaries: list[dict[str, Any]],
    output_path: Path,
    value_field: str,
    y_label: str,
    title: str,
    logarithmic: bool,
    transform: Any | None = None,
) -> None:
    """Draw one method-versus-depth line panel for each target."""
    figure, axes = plt.subplots(1, len(TARGET_SPECS), figsize=(14.8, 4.25), sharex=True)
    for axis, target in zip(axes, TARGET_SPECS):
        target_rows = rows_for_target(summaries, target.key)
        for method_name in METHOD_NAMES:
            method_rows = sorted(
                (row for row in target_rows if row["method"] == method_name),
                key=lambda row: row["layers"],
            )
            values = np.asarray([row[value_field] for row in method_rows], dtype=np.float64)
            if transform is not None:
                values = transform(values)
            axis.plot(
                EXPECTED_LAYERS,
                values,
                color=METHOD_COLORS[method_name],
                marker=METHOD_MARKERS[method_name],
                linewidth=1.8,
                markersize=5,
                label=METHOD_SHORT_LABELS[method_name],
            )
        if logarithmic:
            axis.set_yscale("log")
        axis.set_title(target.short_label)
        axis.set_xlabel("circuit layers $n$")
        axis.set_xticks(EXPECTED_LAYERS)
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(y_label)
    figure.suptitle(title, y=1.02)
    add_shared_legend(figure, axes)
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_paired_speedups(
    summaries: list[dict[str, Any]], output_path: Path
) -> None:
    """Plot median paired timing ratios with interquartile bands."""
    figure, axes = plt.subplots(1, len(TARGET_SPECS), figsize=(14.8, 4.25), sharex=True)
    for axis, target in zip(axes, TARGET_SPECS):
        target_rows = rows_for_target(summaries, target.key)
        for method_name in METHOD_NAMES:
            method_rows = sorted(
                (row for row in target_rows if row["method"] == method_name),
                key=lambda row: row["layers"],
            )
            medians = np.asarray(
                [row["paired_speedup_vs_full_coherent_median"] for row in method_rows]
            )
            lower = np.asarray(
                [row["paired_speedup_vs_full_coherent_q25"] for row in method_rows]
            )
            upper = np.asarray(
                [row["paired_speedup_vs_full_coherent_q75"] for row in method_rows]
            )
            axis.plot(
                EXPECTED_LAYERS,
                medians,
                color=METHOD_COLORS[method_name],
                marker=METHOD_MARKERS[method_name],
                linewidth=1.8,
                markersize=5,
                label=METHOD_SHORT_LABELS[method_name],
            )
            axis.fill_between(
                EXPECTED_LAYERS,
                lower,
                upper,
                color=METHOD_COLORS[method_name],
                alpha=0.12,
            )
        axis.axhline(1.0, color="black", linewidth=0.9, linestyle="--")
        axis.set_yscale("log")
        axis.set_title(target.short_label)
        axis.set_xlabel("circuit layers $n$")
        axis.set_xticks(EXPECTED_LAYERS)
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(r"paired speedup $T_{\rm full}/T_{\rm method}$")
    figure.suptitle("Optimizer speed on matched hardware (median and interquartile range)", y=1.02)
    add_shared_legend(figure, axes)
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_success_rates(
    summaries: list[dict[str, Any]], output_path: Path, threshold: float = 0.99
) -> None:
    """Plot the fraction of restarts reaching a fidelity threshold."""
    suffix = str(threshold).replace(".", "p")
    field = f"success_rate_f_ge_{suffix}"
    low_field = f"{field}_wilson_low"
    high_field = f"{field}_wilson_high"
    figure, axes = plt.subplots(1, len(TARGET_SPECS), figsize=(14.8, 4.25), sharex=True)
    for axis, target in zip(axes, TARGET_SPECS):
        target_rows = rows_for_target(summaries, target.key)
        for method_name in METHOD_NAMES:
            method_rows = sorted(
                (row for row in target_rows if row["method"] == method_name),
                key=lambda row: row["layers"],
            )
            rates = np.asarray([row[field] for row in method_rows])
            lower = np.asarray([row[low_field] for row in method_rows])
            upper = np.asarray([row[high_field] for row in method_rows])
            axis.errorbar(
                EXPECTED_LAYERS,
                rates,
                yerr=np.vstack(
                    (
                        np.maximum(0.0, rates - lower),
                        np.maximum(0.0, upper - rates),
                    )
                ),
                color=METHOD_COLORS[method_name],
                marker=METHOD_MARKERS[method_name],
                linewidth=1.6,
                capsize=2,
                label=METHOD_SHORT_LABELS[method_name],
            )
        axis.set_ylim(-0.03, 1.03)
        axis.set_title(target.short_label)
        axis.set_xlabel("circuit layers $n$")
        axis.set_xticks(EXPECTED_LAYERS)
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel(rf"fraction of restarts with $F\geq {threshold}$")
    figure.suptitle("Restart reliability (95% Wilson intervals)", y=1.02)
    add_shared_legend(figure, axes)
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_infidelity_heatmaps(
    summaries: list[dict[str, Any]], output_path: Path
) -> None:
    """Plot best infidelity as decimal digits of accuracy for every task."""
    figure, axes = plt.subplots(1, len(TARGET_SPECS), figsize=(14.6, 5.0), sharey=True)
    maximum_digits = 0.0
    arrays: dict[str, np.ndarray] = {}
    for target in TARGET_SPECS:
        indexed = {
            (row["method"], row["layers"]): row
            for row in rows_for_target(summaries, target.key)
        }
        values = np.asarray(
            [
                [
                    -math.log10(
                        max(
                            indexed[(method_name, layer_count)]["best_exact_infidelity"],
                            PLOTTING_INFIDELITY_FLOOR,
                        )
                    )
                    for layer_count in EXPECTED_LAYERS
                ]
                for method_name in METHOD_NAMES
            ],
            dtype=np.float64,
        )
        arrays[target.key] = values
        maximum_digits = max(maximum_digits, float(np.max(values)))

    image = None
    for axis, target in zip(axes, TARGET_SPECS):
        values = arrays[target.key]
        image = axis.imshow(values, cmap="viridis", vmin=0.0, vmax=maximum_digits, aspect="auto")
        axis.set_title(target.short_label)
        axis.set_xticks(range(len(EXPECTED_LAYERS)), EXPECTED_LAYERS)
        axis.set_xlabel("circuit layers $n$")
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                text_color = "white" if values[row_index, column_index] < 0.62 * maximum_digits else "black"
                axis.text(
                    column_index,
                    row_index,
                    f"{values[row_index, column_index]:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )
    axes[0].set_yticks(
        range(len(METHOD_NAMES)),
        [METHOD_SHORT_LABELS[method_name] for method_name in METHOD_NAMES],
    )
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
        colorbar.set_label(r"accuracy digits $-\log_{10}(1-F^*)$")
    figure.suptitle("Best exact fidelity among 100 restarts", y=1.01)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def make_plots(summaries: list[dict[str, Any]], figure_directory: Path) -> None:
    """Generate every cross-target comparison figure."""
    figure_directory.mkdir(parents=True, exist_ok=True)
    configure_plots()
    plot_lines_by_target(
        summaries,
        figure_directory / "exact_infidelity_vs_layers.png",
        "best_exact_infidelity",
        r"best exact infidelity $1-F^*$",
        "Best exact state-preparation error among 100 restarts",
        logarithmic=True,
        transform=lambda values: np.maximum(values, PLOTTING_INFIDELITY_FLOOR),
    )
    plot_lines_by_target(
        summaries,
        figure_directory / "total_search_time_vs_layers.png",
        "total_multirestart_wall_seconds",
        "sum of 100 restart times (hours)",
        "Raw optimizer process time (hardware may differ between targets)",
        logarithmic=True,
        transform=lambda values: values / 3600.0,
    )
    plot_lines_by_target(
        summaries,
        figure_directory / "coherax_gate_time_vs_layers.png",
        "coherax_gate_time_microseconds",
        r"Coherax gate time $T_C$ ($\mu$s)",
        "Physical execution-time estimate for the selected circuit",
        logarithmic=False,
    )
    plot_lines_by_target(
        summaries,
        figure_directory / "eickbusch_gate_time_bound_vs_layers.png",
        "eickbusch_lower_bound_microseconds",
        r"Eickbusch lower bound $T_{E,\mathrm{LB}}$ ($\mu$s)",
        "Physical execution-time lower bound for the selected circuit",
        logarithmic=False,
    )
    plot_paired_speedups(summaries, figure_directory / "paired_speedup_vs_full_coherent.png")
    plot_success_rates(summaries, figure_directory / "restart_success_fidelity_0p99.png")
    plot_infidelity_heatmaps(summaries, figure_directory / "best_infidelity_heatmap.png")


# %% Tables and machine-readable artifacts

def atomic_json(path: Path, value: Any) -> None:
    """Write a JSON value atomically."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        json.dump(value, temporary_file, indent=2, allow_nan=False)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)
    os.replace(temporary_path, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write records to CSV with a stable union of field names."""
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(value: str) -> str:
    """Escape plain text for a LaTeX table cell."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def write_report_results(
    output_path: Path,
    summaries: list[dict[str, Any]],
    method_summaries: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
    expected_restart_count: int,
) -> None:
    """Write data-dependent LaTeX tables consumed by ``report.tex``."""
    lines = [
        rf"\newcommand{{\AggregateRestartCount}}{{{expected_restart_count}}}",
        rf"\newcommand{{\AggregateRestartRecords}}{{{len(TARGET_SPECS) * len(METHOD_NAMES) * len(EXPECTED_LAYERS) * expected_restart_count}}}",
        rf"\newcommand{{\AggregateWinnerRecords}}{{{len(summaries)}}}",
        r"\begin{table}[H]",
        r"\centering\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"representation & mean rank & rank-1 tasks & geom. mean $1-F^*$ & median speedup & process h\\",
        r"\midrule",
    ]
    for row in method_summaries:
        lines.append(
            f"{tex_escape(row['method_label'])} & "
            f"{row['mean_fidelity_rank']:.2f} & "
            f"{row['best_of_five_task_count']}/15 & "
            f"{row['geometric_mean_best_infidelity']:.3e} & "
            f"{row['median_fair_speedup_vs_full_coherent']:.2f}$\\times$ & "
            f"{row['total_optimizer_process_hours']:.2f}\\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Descriptive cross-task summary. The geometric mean is not a physical fidelity; it only compresses 15 heterogeneous benchmark errors into one number.}",
            r"\end{table}",
            "",
            r"\begin{table}[H]",
            r"\centering\small",
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            r"target & best representation & depth & exact $F^*$ & $1-F^*$\\",
            r"\midrule",
        ]
    )
    for target in TARGET_SPECS:
        target_rows = rows_for_target(summaries, target.key)
        best = max(target_rows, key=lambda row: row["best_exact_fidelity"])
        lines.append(
            f"{tex_escape(target.target_formula)} & "
            f"{tex_escape(best['method_label'])} & {best['layers']} & "
            f"{best['best_exact_fidelity']:.12f} & "
            f"{best['best_exact_infidelity']:.3e}\\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Best observed circuit over all five representations and five depths for each target.}",
            r"\end{table}",
            "",
            r"\paragraph{Input validation.}",
        ]
    )
    for manifest in manifests:
        lines.append(
            f"{tex_escape(manifest['target'])}: {manifest['individual_restart_record_count']} "
            f"restart records, {manifest['fair_timing_restart_count']}/{expected_restart_count} fair timing blocks, "
            f"SHA-256 \texttt{{{manifest['sha256'][:16]}...}}.\\\\"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def artifact_inventory(repository_root: Path) -> list[dict[str, Any]]:
    """Record whether useful source artifacts accompanied the aggregates."""
    rows = []
    for target in TARGET_SPECS:
        source = repository_root / "experiments" / target.source_directory
        for relative_path in (
            Path("results.json"),
            Path("best_sequences.npz"),
            Path("figs/wigner_optimizer_grid.png"),
            Path("cluster/aggregate_best_sequences.npz"),
            Path("cluster/hardware_summary.csv"),
        ):
            path = source / relative_path
            rows.append(
                {
                    "target": target.key,
                    "artifact": str(relative_path),
                    "present": path.exists(),
                    "size_bytes": path.stat().st_size if path.exists() else None,
                }
            )
    return rows


def run_aggregation(
    repository_root: Path,
    output_directory: Path,
    expected_restart_count: int = EXPECTED_RESTART_COUNT,
    create_plots: bool = True,
) -> dict[str, Any]:
    """Validate all inputs and write the complete Experiment 09 outputs."""
    if expected_restart_count <= 0:
        raise ValueError("expected_restart_count must be positive")
    output_directory.mkdir(parents=True, exist_ok=True)
    sources: dict[str, dict[str, Any]] = {}
    manifests: list[dict[str, Any]] = []
    restart_rows: list[dict[str, Any]] = []
    for target in TARGET_SPECS:
        aggregate, manifest = load_and_validate_source(
            repository_root, target, expected_restart_count
        )
        sources[target.key] = aggregate
        manifests.append(manifest)
        restart_rows.extend(flatten_restarts(target, aggregate))

    expected_individual_records = (
        len(TARGET_SPECS)
        * len(METHOD_NAMES)
        * len(EXPECTED_LAYERS)
        * expected_restart_count
    )
    if len(restart_rows) != expected_individual_records:
        raise ValueError(
            f"flattened {len(restart_rows)} restart rows; expected "
            f"{expected_individual_records}"
        )

    winner_summaries = summarize_winners(sources, expected_restart_count)
    method_summaries = summarize_methods(winner_summaries)
    if create_plots:
        make_plots(winner_summaries, output_directory / "figs")

    inventory = artifact_inventory(repository_root)
    result = {
        "experiment": "09_cross_target_optimizer_aggregation",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_contract": {
            "targets": len(TARGET_SPECS),
            "methods": len(METHOD_NAMES),
            "depths": len(EXPECTED_LAYERS),
            "restarts_per_method_depth_target": expected_restart_count,
            "expected_winner_records": len(winner_summaries),
            "expected_individual_restart_records": expected_individual_records,
            "winner_selection": "maximum cutoff-free exact fidelity within each method/depth/target",
        },
        "definitions": {
            "exact_fidelity": (
                "F = sum_q |<target|psi_q>|^2, evaluated from the saved physical "
                "circuit by the cutoff-free coherent-state overlap"
            ),
            "best_exact_fidelity": "F* = max over the 100 restart exact fidelities",
            "best_exact_infidelity": "1 - F*",
            "total_multirestart_wall_seconds": (
                "sum of optimizer-plus-exact-rescore process wall times over 100 sequential restarts"
            ),
            "paired_speedup_vs_full_coherent": (
                "for the same target, depth, restart, and hardware block: "
                "T_full_coherent / T_method"
            ),
            "restart_success_rate": (
                "number of the 100 restarts with exact fidelity at least a stated "
                "threshold, divided by 100"
            ),
            "restart_win_share": (
                "fractional share of restart indices at which a method has the "
                "largest exact fidelity among the five methods; exact ties split one vote"
            ),
            "geometric_mean_best_infidelity": (
                "exp(mean(log(max(1-F*, 1e-16)))) across 15 heterogeneous tasks; "
                "a descriptive aggregate, not a physical state fidelity"
            ),
        },
        "source_manifests": manifests,
        "artifact_inventory": inventory,
        "winner_summaries": winner_summaries,
        "method_summaries": method_summaries,
    }
    atomic_json(output_directory / "results.json", result)
    write_csv(output_directory / "winner_summary.csv", winner_summaries)
    write_csv(output_directory / "all_restart_results.csv", restart_rows)
    write_csv(output_directory / "method_summary.csv", method_summaries)
    write_csv(output_directory / "source_artifact_inventory.csv", inventory)
    write_report_results(
        output_directory / "report_results.tex",
        winner_summaries,
        method_summaries,
        manifests,
        expected_restart_count,
    )
    return result


# %% Reproduction entry point

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for local or fixture-based aggregation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=repository_root_from_script(),
        help="Checkout containing experiments/06, experiments/07, and experiments/08.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path.cwd(),
        help="Directory receiving Experiment 09 artifacts.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Validate and write tables without rendering figures (useful for tests).",
    )
    return parser.parse_args()


def main() -> None:
    """Run aggregation and print a compact completion record."""
    arguments = parse_arguments()
    result = run_aggregation(
        repository_root=arguments.repository_root.resolve(),
        output_directory=arguments.output_directory.resolve(),
        expected_restart_count=EXPECTED_RESTART_COUNT,
        create_plots=not arguments.skip_plots,
    )
    print(
        "EXPERIMENT_09_AGGREGATION_COMPLETE",
        json.dumps(
            {
                "winner_records": len(result["winner_summaries"]),
                "restart_records": result["input_contract"][
                    "expected_individual_restart_records"
                ],
                "output_directory": str(arguments.output_directory.resolve()),
            }
        ),
        flush=True,
    )


if __name__ == "__main__" and not globals().get("_EXPERIMENT_09_NOTEBOOK", False):
    main()
