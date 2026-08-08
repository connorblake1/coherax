#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$experiment_dir/../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$experiment_dir"

python _build_notebook.py
"$repo_root/scripts/safe_guard.sh" --wall-minutes 420 --memory-gb 12 -- \
  python _run_nb.py

tinytex_pdflatex="/Users/cjblake/Library/TinyTeX/bin/universal-darwin/pdflatex"
"$tinytex_pdflatex" -interaction=nonstopmode -halt-on-error report.tex
"$tinytex_pdflatex" -interaction=nonstopmode -halt-on-error report.tex
"$tinytex_pdflatex" -interaction=nonstopmode -halt-on-error report.tex
