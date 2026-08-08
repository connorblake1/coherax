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
        winning_hardware = winner.get("hardware", {})
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
                "winning_node": winning_hardware.get("node"),
                "winning_cpu_model": winning_hardware.get("cpu_model"),
                "winning_hardware": winning_hardware,
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
                "cluster_parallel": False,
                "wall_time_definition": (
                    "sum of independent restart wall times; array tasks ran sequentially"
                ),
            }
        )

if missing_shards:
    preview = ", ".join(missing_shards[:12])
    raise FileNotFoundError(
        f"Missing {len(missing_shards)} restart shards. First missing: {preview}"
    )

# Array task r normally runs all 25 cases for restart r on one allocation.
# Different restart blocks may use different node and CPU models. Record that
# heterogeneity explicitly so timings can be stratified after the run.
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
    "machine_architecture",
    "operating_system",
)
hardware_by_restart = {}
hardware_groups_by_signature = {}
within_restart_hardware_mismatches = []
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
    signatures = {
        tuple(item.get("hardware", {}).get(field) for field in resource_fields)
        for item in matched
    }
    slurm_job_ids = {
        item.get("slurm", {}).get("job_id") for item in matched
    }
    slurm_array_job_ids = {
        item.get("slurm", {}).get("array_job_id") for item in matched
    }
    slurm_array_task_ids = {
        item.get("slurm", {}).get("array_task_id") for item in matched
    }
    matched_single_hardware = (
        len(nodes) == 1
        and None not in nodes
        and len(signatures) == 1
    )
    restart_hardware = {
        "matched_single_hardware": matched_single_hardware,
        "nodes": sorted(str(node) for node in nodes),
        "slurm_job_ids": sorted(str(job_id) for job_id in slurm_job_ids),
        "slurm_array_job_ids": sorted(
            str(job_id) for job_id in slurm_array_job_ids
        ),
        "slurm_array_task_ids": sorted(
            str(task_id) for task_id in slurm_array_task_ids
        ),
        "resource_signatures": [
            dict(zip(resource_fields, signature))
            for signature in sorted(signatures, key=repr)
        ],
    }
    if matched_single_hardware:
        node = next(iter(nodes))
        signature = next(iter(signatures))
        signature_record = dict(zip(resource_fields, signature))
        restart_hardware.update({"node": node, **signature_record})
        group = hardware_groups_by_signature.setdefault(
            signature,
            {
                "signature": signature_record,
                "restart_indices": [],
                "nodes": set(),
            },
        )
        group["restart_indices"].append(restart_index)
        group["nodes"].add(node)
    else:
        within_restart_hardware_mismatches.append(restart_index)
    hardware_by_restart[restart_index] = restart_hardware

hardware_groups = []
for group in hardware_groups_by_signature.values():
    hardware_groups.append(
        {
            "restart_count": len(group["restart_indices"]),
            "restart_indices": group["restart_indices"],
            "nodes": sorted(group["nodes"]),
            "signature": group["signature"],
        }
    )
hardware_groups.sort(
    key=lambda group: (
        str(group["signature"].get("cpu_model")),
        group["restart_indices"][0],
    )
)
within_restart_hardware_matched = not within_restart_hardware_mismatches
hardware_homogeneous_across_restarts = (
    within_restart_hardware_matched and len(hardware_groups) == 1
)
common_hardware = (
    hardware_groups[0]["signature"]
    if hardware_homogeneous_across_restarts
    else None
)

aggregate = {
    "experiment": EXPERIMENT_DIRECTORY.name,
    "restart_count": arguments.restart_count,
    "record_count": len(records),
    "wall_time_definition": (
        "sum of independent restart wall times; Slurm array tasks ran sequentially"
    ),
    "timing_fairness": {
        "design": (
            "matched blocks: restart r ran all 25 method/depth cases as fresh "
            "processes on one Slurm allocation"
        ),
        "common_hardware": common_hardware,
        "hardware_by_restart": hardware_by_restart,
        "hardware_groups": hardware_groups,
        "hardware_group_count": len(hardware_groups),
        "hardware_homogeneous_across_restarts": (
            hardware_homogeneous_across_restarts
        ),
        "within_restart_hardware_matched": within_restart_hardware_matched,
        "within_restart_hardware_mismatches": (
            within_restart_hardware_mismatches
        ),
        "execution_order": (
            "the 25-case order was cyclically rotated by restart modulo 25"
        ),
        "post_hoc_comparison": (
            "use hardware_summary.csv or each restart record's hardware object "
            "to group wall times by CPU model, node, and software environment"
        ),
        "validated": within_restart_hardware_matched,
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
            "winning_node",
            "winning_cpu_model",
            "exact_fidelity",
            "exact_infidelity",
            "native_fidelity",
            "total_multirestart_wall_seconds",
        ),
    )
    writer.writeheader()
    for record in records:
        writer.writerow({name: record[name] for name in writer.fieldnames})

with (CLUSTER_DIRECTORY / "hardware_summary.csv").open(
    "w", encoding="utf-8", newline=""
) as hardware_summary_file:
    hardware_writer = csv.DictWriter(
        hardware_summary_file,
        fieldnames=(
            "restart",
            "matched_single_hardware",
            "nodes",
            "slurm_job_ids",
            "slurm_array_job_ids",
            "slurm_array_task_ids",
            *resource_fields,
        ),
    )
    hardware_writer.writeheader()
    for restart_index, restart_hardware in hardware_by_restart.items():
        signatures = restart_hardware["resource_signatures"]
        hardware_writer.writerow(
            {
                "restart": restart_index,
                "matched_single_hardware": restart_hardware[
                    "matched_single_hardware"
                ],
                "nodes": ";".join(restart_hardware["nodes"]),
                "slurm_job_ids": ";".join(
                    restart_hardware["slurm_job_ids"]
                ),
                "slurm_array_job_ids": ";".join(
                    restart_hardware["slurm_array_job_ids"]
                ),
                "slurm_array_task_ids": ";".join(
                    restart_hardware["slurm_array_task_ids"]
                ),
                **{
                    field: ";".join(
                        sorted(
                            {
                                str(signature.get(field))
                                for signature in signatures
                            }
                        )
                    )
                    for field in resource_fields
                },
            }
        )

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
    published["configuration"]["cluster_parallel"] = False
    published["configuration"]["cluster_restart_count"] = arguments.restart_count
    published["configuration"]["cluster_timing_fairness"] = aggregate[
        "timing_fairness"
    ]
    published["timing_definitions"]["optimization_wall_time"] = (
        "sum of independent restart wall times; Slurm array tasks ran sequentially"
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
