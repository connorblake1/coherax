"""Run one independently indexed optimizer restart and save an atomic shard."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import tempfile
import time
from pathlib import Path
import sys

import numpy as np

EXPERIMENT_DIRECTORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENT_DIRECTORY))
import _optimization_experiment as experiment


CLUSTER_DIRECTORY = Path(__file__).resolve().parent
RESULT_DIRECTORY = CLUSTER_DIRECTORY / "results"
RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)


def cpu_model_name() -> str:
    """Return a stable CPU model string on Linux compute nodes."""
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        for line in cpuinfo_path.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def hardware_record() -> dict[str, str | None]:
    """Record the hardware/resource identity used for timing validation."""
    allowed_cpu_count = (
        str(len(os.sched_getaffinity(0)))
        if hasattr(os, "sched_getaffinity")
        else None
    )
    return {
        "node": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
        "cpu_model": cpu_model_name(),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "constraint": (
            os.environ.get("SLURM_JOB_CONSTRAINTS")
            or os.environ.get("COHERAX_HARDWARE_CONSTRAINT")
        ),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "allowed_cpu_count": allowed_cpu_count,
        "memory_per_node_mb": os.environ.get("SLURM_MEM_PER_NODE"),
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "jax_version": experiment.jax.__version__,
        "dynamiqs_version": getattr(experiment.dq, "__version__", "unknown"),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
    }


def decode_global_index(
    global_index: int,
    restart_count: int,
) -> tuple[str, int, int]:
    """Map one zero-based array index to method, depth, and restart."""
    total_jobs = (
        len(experiment.METHOD_NAMES)
        * len(experiment.LAYER_COUNTS)
        * restart_count
    )
    if not 0 <= global_index < total_jobs:
        raise ValueError(
            f"global_index must be in [0, {total_jobs}), got {global_index}"
        )
    restart_index = global_index % restart_count
    method_depth_index = global_index // restart_count
    depth_index = method_depth_index % len(experiment.LAYER_COUNTS)
    method_index = method_depth_index // len(experiment.LAYER_COUNTS)
    return (
        experiment.METHOD_NAMES[method_index],
        experiment.LAYER_COUNTS[depth_index],
        restart_index,
    )


def outer_steps_for_depth(layer_count: int) -> int:
    """Use the experiment's canonical depth-dependent update budget."""
    if hasattr(experiment, "DEPTH_TWENTY_OUTER_STEPS") and layer_count == 20:
        return int(experiment.DEPTH_TWENTY_OUTER_STEPS)
    return int(experiment.OUTER_STEPS)


def atomic_json(path: Path, value: dict) -> None:
    """Write JSON completely before replacing the public shard."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        json.dump(value, temporary_file, indent=2)
        temporary_path = Path(temporary_file.name)
    os.replace(temporary_path, path)


def atomic_npz(path: Path, parameters: np.ndarray) -> None:
    """Write an NPZ parameter shard completely before publishing it."""
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        np.savez(temporary_file, real_parameters=parameters)
        temporary_path = Path(temporary_file.name)
    os.replace(temporary_path, path)


parser = argparse.ArgumentParser()
parser.add_argument("--global-index", type=int, required=True)
parser.add_argument("--restart-count", type=int, required=True)
parser.add_argument("--force", action="store_true")
arguments = parser.parse_args()

if arguments.restart_count != 100:
    raise ValueError("this matched cluster experiment requires exactly 100 restarts")

method_name, layer_count, restart_index = decode_global_index(
    arguments.global_index,
    arguments.restart_count,
)
stem = f"{method_name}_n{layer_count}_r{restart_index:05d}"
json_path = RESULT_DIRECTORY / f"{stem}.json"
npz_path = RESULT_DIRECTORY / f"{stem}.npz"

if json_path.exists() and npz_path.exists() and not arguments.force:
    print("CLUSTER_RESTART_SKIPPED", stem, flush=True)
    raise SystemExit(0)

initial_parameters = experiment.deterministic_initial_parameters(
    layer_count,
    restart_index,
)
fidelity_function = experiment.fidelity_function_for_method(method_name)
outer_steps = outer_steps_for_depth(layer_count)

started = time.perf_counter()
optimized_parameters, native_loss = experiment.run_optimizer_restart(
    initial_parameters,
    fidelity_function,
    outer_steps=outer_steps,
)
optimizer_wall_seconds = time.perf_counter() - started
exact_started = time.perf_counter()
exact_fidelity = float(
    experiment.exact_coherent_score_numpy(optimized_parameters)
)
exact_rescore_wall_seconds = time.perf_counter() - exact_started

record = {
    "global_index": arguments.global_index,
    "method": method_name,
    "method_label": experiment.METHOD_LABELS[method_name],
    "layers": layer_count,
    "restart": restart_index,
    "configured_restart_count": arguments.restart_count,
    "outer_steps": outer_steps,
    "native_fidelity": float(np.clip(1.0 - native_loss, 0.0, 1.0)),
    "exact_fidelity": exact_fidelity,
    "exact_infidelity": 1.0 - exact_fidelity,
    "optimizer_wall_seconds": optimizer_wall_seconds,
    "exact_rescore_wall_seconds": exact_rescore_wall_seconds,
    "wall_seconds": optimizer_wall_seconds + exact_rescore_wall_seconds,
    "total_wall_seconds": optimizer_wall_seconds + exact_rescore_wall_seconds,
    "coherax_gate_time_microseconds": (
        experiment.coherax_gate_time_microseconds(optimized_parameters)
    ),
    "eickbusch_lower_bound_microseconds": (
        experiment.eickbusch_gate_time_lower_bound_microseconds(
            optimized_parameters
        )
    ),
    "slurm": {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
    },
    "hardware": hardware_record(),
}

atomic_npz(npz_path, np.asarray(optimized_parameters, dtype=np.float64))
atomic_json(json_path, record)
print("CLUSTER_RESTART_COMPLETE", json.dumps(record), flush=True)
