# coherax

Coherent-basis optimization toolkit for bosonic quantum error-correcting codes.
Provides analytic closed-form fidelity formulas, CD+R circuit optimization,
transpose channel recovery, floating-basis codes, and coherent information benchmarks.

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
# Run the notebook
jupyter notebook aggregated_results.ipynb    # select "Python (coherax)" kernel

# Run tests
pytest coherax/verification.py -v

# Run individual scripts
python -m coherax.benchmark_claude          # GKP recovery benchmark
python fock_fidelity_claude.py                # Fock state preparation optimization
python benchmark_codes_claude.py              # Code comparison (cat, binomial, GKP)
```

## Project Structure

```
coherax/
├── aggregated_results.ipynb      # Main results notebook (8 sections)
├── benchmark_codes_claude.py     # Cat/binomial/GKP code generators & benchmarks
├── fock_fidelity_claude.py       # Analytic Fock fidelity & CD+R circuit optimization
├── coherax/                    # Core library
│   ├── characteristic_jax_utils.py   # CoherentKet, BosonicSubspace, loss channels, JAX ops
│   ├── transpose_channel_claude.py   # Transpose recovery, SBS baseline, F_e computation
│   ├── worstcase_optimizer_claude.py # CMA-ES worst-case fidelity optimization
│   ├── utils.py                      # GKP constants, quantum operators, Kraus maps
│   ├── analytic_utils.py             # Symbolic circuit representations
│   ├── jax_analytic_utils.py         # JAX circuit layers (JLayer/Equinox)
│   ├── verification.py               # Pytest test suite
│   ├── fourier_saved.npy             # Precomputed Fourier coefficients (committed)
│   └── ...                           # Optimizer scripts, sweep scripts, etc.
└── data/                          # Saved results (.npz files) — see below
```

## Data Files

The notebook loads `.npz` result files. These must be placed in `data/` relative
to the repo root. The notebook's `BASE` variable should point here.

Files needed (copy from `jiang-research/FiniteGKP/`):
- `data/fock_preparation.npz` — from `fock_preparation/results/fock_preparation.npz`
- `data/cmaes_recovery_params.npz` — from `results/cmaes_recovery_params.npz`
- `data/extended_depth_sweep.npz` — from `results/extended_depth_sweep.npz`
- `data/entanglement_fidelity_benchmark.npz` — from `results/entanglement_fidelity_benchmark.npz`

## Key Constants

- `GKP_N = 100` — Fock space truncation dimension
- `GKP_L = 2*sqrt(pi)` — GKP lattice spacing
- `Delta` — GKP envelope parameter (typically 0.2–0.3)

## Notes

- All JAX code uses `jax.config.update("jax_enable_x64", True)` for double precision
- The `_claude` suffix on filenames is a provenance marker (Claude-assisted authoring)
- StrawberryFields is only used in `coherax/utils.py` for GKP Fock-basis state generation
