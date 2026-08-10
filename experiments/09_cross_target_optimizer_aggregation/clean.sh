#!/usr/bin/env bash
# Remove only artifacts generated inside Experiment 09.
set -euo pipefail

experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
notebook_name="cross_target_optimizer_aggregation.ipynb"
dry_run=false

if [[ $# -gt 1 || (${1:-} != "" && ${1:-} != "--dry-run") ]]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi
if [[ ${1:-} == "--dry-run" ]]; then
  dry_run=true
fi
if [[ ! -f "$experiment_dir/_aggregate_results.py" || ! -f "$experiment_dir/report.tex" ]]; then
  echo "clean.sh: Experiment 09 source files are missing; refusing to clean" >&2
  exit 2
fi

shopt -s nullglob
generated_files=(
  "$experiment_dir/results.json"
  "$experiment_dir/all_restart_results.csv"
  "$experiment_dir/winner_summary.csv"
  "$experiment_dir/method_summary.csv"
  "$experiment_dir/source_artifact_inventory.csv"
  "$experiment_dir/report_results.tex"
  "$experiment_dir/report.pdf"
  "$experiment_dir/report.aux"
  "$experiment_dir/report.log"
  "$experiment_dir/report.out"
  "$experiment_dir/report.fls"
  "$experiment_dir/report.fdb_latexmk"
  "$experiment_dir/report.synctex.gz"
  "$experiment_dir/figs/"*.png
  "$experiment_dir/figs/"*.pdf
  "$experiment_dir/.matplotlib/"fontlist*.json
  "$experiment_dir/__pycache__/"*.pyc
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
