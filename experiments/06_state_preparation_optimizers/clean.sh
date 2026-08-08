#!/usr/bin/env bash
# Remove generated Experiment 06 outputs while preserving all source files.
set -euo pipefail

experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
notebook_name="state_preparation_optimizers.ipynb"
dry_run=false

if [[ $# -gt 1 || (${1:-} != "" && ${1:-} != "--dry-run") ]]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi
if [[ ${1:-} == "--dry-run" ]]; then
  dry_run=true
fi

# Refuse to operate if this file has been moved outside an experiment folder.
if [[ ! -f "$experiment_dir/_optimization_experiment.py" || ! -d "$experiment_dir/cluster" ]]; then
  echo "clean.sh: expected experiment source files are missing; refusing to clean" >&2
  exit 2
fi

shopt -s nullglob
generated_files=(
  "$experiment_dir/results.json"
  "$experiment_dir/best_sequences.npz"
  "$experiment_dir/report.pdf"
  "$experiment_dir/report.aux"
  "$experiment_dir/report.log"
  "$experiment_dir/report.out"
  "$experiment_dir/report.toc"
  "$experiment_dir/report.fls"
  "$experiment_dir/report.fdb_latexmk"
  "$experiment_dir/report.synctex.gz"
  "$experiment_dir/."$notebook_name.*.tmp
  "$experiment_dir/cluster/aggregate_results.json"
  "$experiment_dir/cluster/aggregate_summary.csv"
  "$experiment_dir/cluster/aggregate_best_sequences.npz"
  "$experiment_dir/cluster/hardware_summary.csv"
  "$experiment_dir/figs/"*.png
  "$experiment_dir/figs/"*.pdf
  "$experiment_dir/sequences/"*.json
  "$experiment_dir/sequences/"*.npy
  "$experiment_dir/sequences/"*.npz
  "$experiment_dir/cluster/results/"*.json
  "$experiment_dir/cluster/results/"*.npz
  "$experiment_dir/cluster/results/".*.tmp
  "$experiment_dir/cluster/logs/"*.out
  "$experiment_dir/cluster/logs/"*.err
  "$experiment_dir/cluster/logs/"*.log
  "$experiment_dir/tmp/pdfs/"*.png
  "$experiment_dir/.matplotlib/"fontlist*.json
  "$experiment_dir/__pycache__/"*.pyc
  "$experiment_dir/cluster/__pycache__/"*.pyc
)

removed_count=0
for generated_file in "${generated_files[@]}"; do
  if [[ ! -f "$generated_file" && ! -L "$generated_file" ]]; then
    continue
  fi
  relative_name="${generated_file#"$experiment_dir/"}"
  if $dry_run; then
    printf 'would remove %s\n' "$relative_name"
  else
    rm -- "$generated_file"
    printf 'removed %s\n' "$relative_name"
  fi
  removed_count=$((removed_count + 1))
done

# Keep the self-contained notebook source, but remove stale execution output.
notebook_path="$experiment_dir/$notebook_name"
if [[ -f "$notebook_path" ]]; then
  if $dry_run; then
    printf 'would clear execution output from %s\n' "$notebook_name"
  else
    python - "$notebook_path" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

notebook_path = Path(sys.argv[1])
notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
changed = False
for cell in notebook.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    if cell.get("outputs"):
        cell["outputs"] = []
        changed = True
    if cell.get("execution_count") is not None:
        cell["execution_count"] = None
        changed = True
if changed:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=notebook_path.parent,
        prefix=f".{notebook_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        json.dump(notebook, temporary_file, indent=1, ensure_ascii=False)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)
    os.replace(temporary_path, notebook_path)
PY
    printf 'cleared execution output from %s\n' "$notebook_name"
  fi
fi

if $dry_run; then
  printf 'dry run complete: %d generated files would be removed\n' "$removed_count"
else
  printf 'clean complete: removed %d generated files\n' "$removed_count"
fi
