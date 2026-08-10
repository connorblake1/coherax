#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$experiment_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$experiment_dir"

python _build_notebook.py
"$repo_root/scripts/safe_guard.sh" --wall-minutes 30 --memory-gb 8 -- \
  python _run_nb.py

if command -v pdflatex >/dev/null 2>&1; then
  latex_command="$(command -v pdflatex)"
elif [[ -x /Users/cjblake/Library/TinyTeX/bin/universal-darwin/pdflatex ]]; then
  latex_command=/Users/cjblake/Library/TinyTeX/bin/universal-darwin/pdflatex
else
  echo "Aggregation succeeded; pdflatex was not found, so report.pdf was not built." >&2
  exit 0
fi

"$latex_command" -interaction=nonstopmode -halt-on-error report.tex
"$latex_command" -interaction=nonstopmode -halt-on-error report.tex
