"""Aggregate independent restart shards and optionally publish artifacts."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np


CLUSTER_DIRECTORY = Path(__file__).resolve().parent
EXPERIMENT_DIRECTORY = CLUSTER_DIRECTORY.parent
RESULT_DIRECTORY = CLUSTER_DIRECTORY / "results"

parser = argparse.ArgumentParser()
parser.add_argument("--restart-count", type=int, required=True)
parser.add_argument(
    "--publish",
    action="store_true",
    help="Also regenerate canonical results.json, plots, sequences, and NPZ.",
)
arguments = parser.parse_args()
if arguments.restart_count != 100:
    raise ValueError("this matched cluster experiment requires exactly 100 restarts")

experiment_number = EXPERIMENT_DIRECTORY.name.split("_", 1)[0]
os.environ[f"COHERAX_EXP{experiment_number}_RESTARTS"] = str(
    arguments.restart_count
)
sys.path.insert(0, str(EXPERIMENT_DIRECTORY))
experiment = importlib.import_module("_optimization_experiment")

records = []
winning_parameters = {}
missing_shards = []
all_restart_records = []
for method_name in experiment.METHOD_NAMES:
    for layer_count in experiment.LAYER_COUNTS:
        restart_records = []
        parameter_arrays = {}
        for restart_index in range(arguments.restart_count):
            stem = f"{method_name}_n{layer_count}_r{restart_index:05d}"
            json_path = RESULT_DIRECTORY / f"{stem}.json"
            npz_path = RESULT_DIRECTORY / f"{stem}.npz"
            if not json_path.exists() or not npz_path.exists():
                missing_shards.append(stem)
                continue
            with json_path.open(encoding="utf-8") as result_file:
                restart_record = json.load(result_file)
            with np.load(npz_path) as archive:
                parameter_arrays[restart_index] = np.asarray(
                    archive["real_parameters"]
                )
            restart_records.append(restart_record)
            all_restart_records.append(restart_record)
        if len(restart_records) != arguments.restart_count:
            continue
        winner = max(restart_records, key=lambda item: item["exact_fidelity"])
        winner_parameters = parameter_arrays[winner["restart"]]
        winning_parameters[(method_name, layer_count)] = winner_parameters
        records.append(
            {
                "method": method_name,
                "method_label": experiment.METHOD_LABELS[method_name],
                "layers": layer_count,
                "restart_count": arguments.restart_count,
                "outer_steps_per_restart": winner["outer_steps"],
                "total_multirestart_wall_seconds": float(
                    sum(item["total_wall_seconds"] for item in restart_records)
                ),
                "winning_restart": winner["restart"],
                "winning_restart_wall_seconds": winner["total_wall_seconds"],
                "native_fidelity": winner["native_fidelity"],
                "exact_fidelity": winner["exact_fidelity"],
                "exact_infidelity": winner["exact_infidelity"],
                "coherax_gate_time_microseconds": winner[
                    "coherax_gate_time_microseconds"
                ],
                "eickbusch_lower_bound_microseconds": winner[
                    "eickbusch_lower_bound_microseconds"
                ],
                "restarts": restart_records,
                "cluster_parallel": True,
                "wall_time_definition": (
                    "sum of independent restart wall times; tasks ran in parallel"
                ),
            }
        )

if missing_shards:
    preview = ", ".join(missing_shards[:12])
    raise FileNotFoundError(
        f"Missing {len(missing_shards)} restart shards. First missing: {preview}"
    )

# Fair timing requires a matched design. Array task r runs all 25 cases for
# restart r on one allocation. Every restart must use one physical node, and
# all 100 allocations must report the same resource/CPU signature.
resource_fields = (
    "cpu_model",
    "partition",
    "constraint",
    "cpus_per_task",
    "allowed_cpu_count",
    "memory_per_node_mb",
    "omp_threads",
    "mkl_threads",
    "openblas_threads",
    "jax_platforms",
    "python_version",
    "numpy_version",
    "jax_version",
    "dynamiqs_version",
    "conda_environment",
)
hardware_by_restart = {}
for restart_index in range(arguments.restart_count):
    matched = [
        item for item in all_restart_records
        if item["restart"] == restart_index
    ]
    if len(matched) != len(experiment.METHOD_NAMES) * len(experiment.LAYER_COUNTS):
        raise ValueError(
            f"Restart {restart_index} has {len(matched)} records, expected 25"
        )
    nodes = {item.get("hardware", {}).get("node") for item in matched}
    if len(nodes) != 1 or None in nodes:
        raise ValueError(
            f"Restart {restart_index} did not run all 25 cases on one node: {nodes}"
        )
    signatures = {
        tuple(item.get("hardware", {}).get(field) for field in resource_fields)
        for item in matched
    }
    if len(signatures) != 1:
        raise ValueError(
            f"Restart {restart_index} used inconsistent hardware resources"
        )
    hardware_by_restart[restart_index] = {
        "node": next(iter(nodes)),
        **dict(zip(resource_fields, next(iter(signatures)))),
    }

cross_restart_signatures = {
    tuple(record[field] for field in resource_fields)
    for record in hardware_by_restart.values()
}
if len(cross_restart_signatures) != 1:
    raise ValueError(
        "The 100 restart allocations did not use one common CPU/resource signature"
    )
common_hardware = dict(
    zip(resource_fields, next(iter(cross_restart_signatures)))
)

aggregate = {
    "experiment": EXPERIMENT_DIRECTORY.name,
    "restart_count": arguments.restart_count,
    "record_count": len(records),
    "wall_time_definition": (
        "sum of independent restart wall times; Slurm tasks ran in parallel"
    ),
    "timing_fairness": {
        "design": (
            "matched blocks: restart r ran all 25 method/depth cases as fresh "
            "processes on one Slurm allocation"
        ),
        "common_hardware": common_hardware,
        "hardware_by_restart": hardware_by_restart,
        "execution_order": (
            "the 25-case order was cyclically rotated by restart modulo 25"
        ),
        "validated": True,
    },
    "records": records,
}
with (CLUSTER_DIRECTORY / "aggregate_results.json").open(
    "w", encoding="utf-8"
) as aggregate_file:
    json.dump(aggregate, aggregate_file, indent=2)

archive = {}
for (method_name, layer_count), parameters in winning_parameters.items():
    archive[f"{method_name}_n{layer_count}_real_parameters"] = parameters
np.savez(CLUSTER_DIRECTORY / "aggregate_best_sequences.npz", **archive)

with (CLUSTER_DIRECTORY / "aggregate_summary.csv").open(
    "w", encoding="utf-8", newline=""
) as summary_file:
    writer = csv.DictWriter(
        summary_file,
        fieldnames=(
            "method",
            "layers",
            "winning_restart",
            "exact_fidelity",
            "exact_infidelity",
            "native_fidelity",
            "total_multirestart_wall_seconds",
        ),
    )
    writer.writeheader()
    for record in records:
        writer.writerow({name: record[name] for name in writer.fieldnames})

if arguments.publish:
    for record in records:
        key = (record["method"], record["layers"])
        sequence_path = (
            experiment.SEQUENCE_DIRECTORY
            / f"{record['method']}_n{record['layers']}.json"
        )
        with sequence_path.open("w", encoding="utf-8") as sequence_file:
            json.dump(
                {
                    "method": record["method"],
                    "layers": record["layers"],
                    "exact_fidelity": record["exact_fidelity"],
                    "winning_restart": record["winning_restart"],
                    "cluster_restart_count": arguments.restart_count,
                    "sequence": experiment.sequence_json(
                        winning_parameters[key]
                    ),
                },
                sequence_file,
                indent=2,
            )
    experiment.write_experiment_artifacts(records, winning_parameters)
    results_path = EXPERIMENT_DIRECTORY / "results.json"
    with results_path.open(encoding="utf-8") as results_file:
        published = json.load(results_file)
    published["configuration"]["cluster_parallel"] = True
    published["configuration"]["cluster_restart_count"] = arguments.restart_count
    published["configuration"]["cluster_timing_fairness"] = aggregate[
        "timing_fairness"
    ]
    published["timing_definitions"]["optimization_wall_time"] = (
        "sum of independent restart wall times; Slurm tasks ran in parallel"
    )
    with results_path.open("w", encoding="utf-8") as results_file:
        json.dump(published, results_file, indent=2)

print(
    "CLUSTER_AGGREGATION_COMPLETE",
    json.dumps(
        {
            "restart_count": arguments.restart_count,
            "records": len(records),
            "published": arguments.publish,
        }
    ),
    flush=True,
)
