# coherax — Development Notes

## Environment Setup

```bash
conda create -n coherax python=3.11 -y
conda activate coherax
pip install -e ".[dev]"
```

Or manually:
```bash
pip install \
  "jax[cpu]>=0.4.30,<0.7" \
  "dynamiqs>=0.3,<0.4" \
  "optax>=0.2,<0.3" \
  "equinox>=0.11,<0.14" \
  "jaxtyping>=0.2" \
  "matplotlib>=3.8" \
  "numpy>=1.26,<2.3" \
  "scipy>=1.12,<1.14" \
  "sympy>=1.12" \
  "strawberryfields==0.23.0" \
  "cma>=3.3" \
  ipykernel jupyter
python -m ipykernel install --user --name coherax --display-name "Python (coherax)"
```

**scipy must be pinned to <1.14** because StrawberryFields 0.23 imports `scipy.integrate.simps`
which was removed in scipy 1.14.

## Running Code

All commands should be run from the repo root with `conda activate coherax`.

```bash
jupyter notebook aggregated_results.ipynb    # select "Python (coherax)" kernel
python fock_fidelity_claude.py               # Fock state preparation optimization
python benchmark_codes_claude.py             # Code comparison (cat, binomial, GKP)
```

## Library Structure (coherax/)

The library is organized into 5 focused modules:

- `operators.py` — dynamiqs wrappers, constants (GKP_N, sigma matrices, a_op, etc.), channels (pure loss, transpose recovery, Kraus maps), linear algebra helpers
- `states.py` — CoherentKet, CoherentDM, BosonicSubspace classes
- `circuits.py` — CD/ECD/rotation unitaries, TraceoutLayer, g(), channel_from_b(), circuit timing
- `fidelity.py` — analytic fidelity computations (single, batched, with loss recovery)
- `gkp.py` — GKP code state generators (square/rectangular lattice)

Supporting modules:
- `transpose_channel_claude.py` — transpose recovery, SBS baseline, F_e computation
- `worstcase_optimizer_claude.py` — CMA-ES worst-case optimization (lazy-imports coherent_tree_optimizer_claude)
- `coherent_tree_optimizer_claude.py` — tree-structured optimization
- `binary_tree_utils.py` — binary tree Kraus structure
- `characteristic_jax_utils.py` — backward-compat shim (re-exports everything from new modules)

Everything else is in `coherax/deprecated/`.

## Data Files (testing_data/)

The notebook loads `.npz` result files from `testing_data/`:
- `exp_C1_x3_100restart.npz`, `exp_C2_x4_100restart.npz` — 1D marginal prep params
- `fock_preparation.npz` — Fock state preparation results
- `cmaes_recovery_params.npz`, `extended_depth_sweep.npz` — CMA-ES recovery params
- `entanglement_fidelity_benchmark.npz` — benchmark comparison data
- `results_vacuum.npz`, `results_Ic_improvements.npz` — floating-basis results
- `floating_prep_results.npz`, `Ic_comparison_results.npz` — I_c comparison data
- `fourier_saved.npy` — precomputed Fourier coefficients

## Key Constants

- `GKP_N = 100` — Fock space truncation dimension
- `GKP_L = 2*sqrt(pi)` — GKP lattice spacing
- `Delta` — GKP envelope parameter (typically 0.2–0.3)

## Notes

- All JAX code uses `jax.config.update("jax_enable_x64", True)` for double precision
- The `_claude` suffix on filenames is a provenance marker (Claude-assisted authoring)
- StrawberryFields is only used in `coherax/deprecated/utils.py` for GKP Fock-basis state generation

## TODO

- [x] Aggressively lint all active files: remove dead imports, unused functions, commented-out code
- [x] Lint `coherax/deprecated/` — deleted 8 redundant files, kept 28 with unique experimental code
- [x] `coherent_tree_optimizer_claude.py` and `binary_tree_utils.py` moved to deprecated/ (only reachable via lazy import in __main__ block)
- [ ] Connect ReadTheDocs to GitHub repo via readthedocs.org dashboard
