"""Execute the self-contained Experiment 08 notebook."""

from __future__ import annotations

import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


NOTEBOOK = "cat_state_preparation_optimizers.ipynb"
notebook = nbformat.read(NOTEBOOK, as_version=4)
client = NotebookClient(notebook, timeout=43200, kernel_name="coherax")

started = time.time()
with client.setup_kernel():
    print(f"[{time.time() - started:7.1f}s] kernel ready", flush=True)
    code_count = 0
    for cell_index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        code_count += 1
        first_line = cell.source.splitlines()[0][:78] if cell.source else ""
        print(
            f"[{time.time() - started:7.1f}s] -> cell {cell_index}: {first_line}",
            flush=True,
        )
        try:
            client.execute_cell(cell, cell_index)
        except CellExecutionError:
            print(f"[{time.time() - started:7.1f}s] !! cell {cell_index} failed", flush=True)
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    print("\n".join(output.get("traceback", [])), flush=True)
            nbformat.write(notebook, NOTEBOOK)
            sys.exit(1)
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                print(output.get("text", "").rstrip(), flush=True)

nbformat.write(notebook, NOTEBOOK)
print(f"[{time.time() - started:7.1f}s] complete; wrote {code_count} code cells", flush=True)
