"""Tests for the strict cross-target cluster-result aggregator."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def load_aggregation_module() -> ModuleType:
    """Load the ignored experiment helper from its filesystem path."""
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "09_cross_target_optimizer_aggregation"
        / "_aggregate_results.py"
    )
    spec = importlib.util.spec_from_file_location("experiment_09_aggregation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AGGREGATION = load_aggregation_module()


def write_synthetic_source(
    repository_root: Path,
    source_directory: str,
    target_index: int,
    restart_count: int,
) -> Path:
    """Write one complete, internally consistent source aggregate."""
    records = []
    for method_index, method_name in enumerate(AGGREGATION.METHOD_NAMES):
        for layer_count in AGGREGATION.EXPECTED_LAYERS:
            restarts = []
            for restart_index in range(restart_count):
                fidelity = min(
                    0.999999,
                    0.45
                    + 0.02 * target_index
                    + 0.03 * method_index
                    + 0.02 * (layer_count / 4)
                    + 0.001 * restart_index,
                )
                duration = (
                    1.0
                    + target_index
                    + method_index
                    + layer_count / 5
                    + restart_index / 10
                )
                restarts.append(
                    {
                        "restart": restart_index,
                        "outer_steps": 10_000,
                        "native_fidelity": fidelity - 1e-5,
                        "exact_fidelity": fidelity,
                        "exact_infidelity": 1.0 - fidelity,
                        "optimizer_wall_seconds": duration - 0.01,
                        "exact_rescore_wall_seconds": 0.01,
                        "total_wall_seconds": duration,
                        "hardware": {
                            "node": f"node{restart_index}",
                            "cpu_model": "synthetic cpu",
                            "partition": "test",
                            "cpus_per_task": "4",
                            "python_version": "3.11",
                            "numpy_version": "2.0",
                            "jax_version": "0.6.2",
                            "dynamiqs_version": "0.3.4",
                            "conda_environment": "coherax",
                        },
                    }
                )
            winner = max(restarts, key=lambda item: item["exact_fidelity"])
            records.append(
                {
                    "method": method_name,
                    "method_label": AGGREGATION.METHOD_LABELS[method_name],
                    "layers": layer_count,
                    "restart_count": restart_count,
                    "winning_restart": winner["restart"],
                    "exact_fidelity": winner["exact_fidelity"],
                    "exact_infidelity": winner["exact_infidelity"],
                    "coherax_gate_time_microseconds": (
                        layer_count * 0.2 + method_index * 0.01
                    ),
                    "eickbusch_lower_bound_microseconds": (
                        layer_count * 0.3 + method_index * 0.01
                    ),
                    "total_multirestart_wall_seconds": sum(
                        item["total_wall_seconds"] for item in restarts
                    ),
                    "restarts": restarts,
                }
            )
    hardware_by_restart = {
        str(restart_index): {
            "matched_single_hardware": True,
            "node": f"node{restart_index}",
            "nodes": [f"node{restart_index}"],
            "resource_signatures": [{"cpu_model": "synthetic cpu"}],
        }
        for restart_index in range(restart_count)
    }
    aggregate = {
        "experiment": source_directory,
        "restart_count": restart_count,
        "record_count": len(records),
        "timing_fairness": {
            "hardware_by_restart": hardware_by_restart,
            "within_restart_hardware_matched": True,
            "hardware_group_count": 1,
        },
        "records": records,
    }
    path = (
        repository_root
        / "experiments"
        / source_directory
        / "cluster"
        / "aggregate_results.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(aggregate), encoding="utf-8")
    return path


def write_all_sources(repository_root: Path, restart_count: int) -> list[Path]:
    """Write the three source aggregates expected by Experiment 09."""
    return [
        write_synthetic_source(
            repository_root,
            target.source_directory,
            target_index,
            restart_count,
        )
        for target_index, target in enumerate(AGGREGATION.TARGET_SPECS)
    ]


def test_complete_aggregation_writes_all_tables_and_plots(tmp_path: Path) -> None:
    """A complete source set produces the expected record counts and files."""
    restart_count = 3
    write_all_sources(tmp_path, restart_count)
    output = tmp_path / "output"
    result = AGGREGATION.run_aggregation(
        repository_root=tmp_path,
        output_directory=output,
        expected_restart_count=restart_count,
        create_plots=True,
    )

    assert result["input_contract"]["expected_winner_records"] == 75
    assert result["input_contract"]["expected_individual_restart_records"] == 225
    assert len(result["winner_summaries"]) == 75
    assert len(result["method_summaries"]) == 5
    assert all(
        manifest["fair_timing_restart_count"] == restart_count
        for manifest in result["source_manifests"]
    )
    assert len((output / "all_restart_results.csv").read_text().splitlines()) == 226
    expected_figures = {
        "exact_infidelity_vs_layers.png",
        "total_search_time_vs_layers.png",
        "coherax_gate_time_vs_layers.png",
        "eickbusch_gate_time_bound_vs_layers.png",
        "paired_speedup_vs_full_coherent.png",
        "restart_success_fidelity_0p99.png",
        "best_infidelity_heatmap.png",
    }
    assert {path.name for path in (output / "figs").glob("*.png")} == expected_figures


def test_missing_restart_is_rejected(tmp_path: Path) -> None:
    """The aggregator refuses a source whose nested restart list is partial."""
    restart_count = 3
    source_paths = write_all_sources(tmp_path, restart_count)
    damaged = json.loads(source_paths[0].read_text(encoding="utf-8"))
    damaged["records"][0]["restarts"].pop()
    source_paths[0].write_text(json.dumps(damaged), encoding="utf-8")

    with pytest.raises(ValueError, match="restart records; expected 3"):
        AGGREGATION.run_aggregation(
            repository_root=tmp_path,
            output_directory=tmp_path / "output",
            expected_restart_count=restart_count,
            create_plots=False,
        )
