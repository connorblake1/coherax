# coherax — Development Notes

## Environment Setup

```bash
conda create -n coherax python=3.11 -y
conda activate coherax
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

## Active Modules (coherax/)

Only 5 modules are in the live dependency tree:

- `characteristic_jax_utils.py` — core (self-contained): CoherentKet, BosonicSubspace, loss channels, JAX ops
- `transpose_channel_claude.py` — transpose recovery, SBS baseline, F_e computation (imports characteristic_jax_utils)
- `worstcase_optimizer_claude.py` — CMA-ES worst-case optimization (imports characteristic_jax_utils; lazy-imports coherent_tree_optimizer_claude)
- `coherent_tree_optimizer_claude.py` — tree-structured optimization (imports characteristic_jax_utils + binary_tree_utils)
- `binary_tree_utils.py` — binary tree Kraus structure (leaf dependency)

Everything else is in `coherax/deprecated/`.

## Data Files (testing_data/)

The notebook loads `.npz` result files from `testing_data/`.

Files needed (copy from `jiang-research/`):
- `exp_C1_x3_100restart.npz` — from `QSP_replication/`
- `exp_C2_x4_100restart.npz` — from `QSP_replication/`
- `fock_preparation.npz` — from `FiniteGKP/fock_preparation/results/`
- `cmaes_recovery_params.npz` — from `FiniteGKP/results/`
- `extended_depth_sweep.npz` — from `FiniteGKP/results/`
- `entanglement_fidelity_benchmark.npz` — from `FiniteGKP/results/`
- `results_vacuum.npz` — from `floating_basis/`
- `results_Ic_improvements.npz` — from `floating_basis/`
- `floating_prep_results.npz` — from `QSP_replication/`
- `Ic_comparison_results.npz` — from `QSP_replication/`
- `fourier_saved.npy` — from `FiniteGKP/gkp_utils/` (precomputed Fourier coefficients)

## Key Constants

- `GKP_N = 100` — Fock space truncation dimension
- `GKP_L = 2*sqrt(pi)` — GKP lattice spacing
- `Delta` — GKP envelope parameter (typically 0.2–0.3)

## Notes

- All JAX code uses `jax.config.update("jax_enable_x64", True)` for double precision
- The `_claude` suffix on filenames is a provenance marker (Claude-assisted authoring)
- StrawberryFields is only used in `coherax/deprecated/utils.py` for GKP Fock-basis state generation
- `fourier_saved.npy` is loaded by `coherax/deprecated/utils.py` — path updated to use `testing_data/`

## TODO

- [ ] Aggressively lint all active files: remove dead imports, unused functions, commented-out code, and unreachable branches
- [ ] Lint `coherax/deprecated/` files similarly before deciding what to permanently delete vs keep
- [ ] Update notebook to load from `testing_data/` and save figures to `figs/` (still uses hardcoded jiang-research BASE path)
- [ ] Decide whether `coherent_tree_optimizer_claude.py` and `binary_tree_utils.py` should also be deprecated (only reachable via a lazy import in worstcase_optimizer_claude.py)
- [ ] Add `pyproject.toml` or `setup.py` for proper packaging
- [ ] Add `.gitignore` for `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`
- [ ] Choose a license
