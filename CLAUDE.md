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
jupyter notebook demo.ipynb    # select "Python (coherax)" kernel
```

## Library Structure (coherax/)

The library is organized into 7 modules:

- `operators.py` — dynamiqs wrappers, constants (GKP_N, sigma matrices, a_op, etc.), channels (pure loss, thermal loss, transpose recovery, Kraus maps), linear algebra helpers
- `states.py` — CoherentKet, CoherentDM, BosonicSubspace classes
- `circuits.py` — CD/ECD/rotation unitaries, TraceoutLayer, g(), channel_from_b(), circuit timing
- `fidelity.py` — analytic fidelity computations (single, batched, Fock state targets)
- `gkp.py` — GKP code state generators (square/rectangular lattice)
- `info.py` — coherent information computations (pure loss, thermal loss)
- `optimizers.py` — gradient-based ECD circuit optimization (state prep and state transfer)

## Data Files (testing_data/)

The demo notebook loads `.npz` result files from `testing_data/`:
- `exp_C1_x3_100restart.npz`, `exp_C2_x4_100restart.npz` — 1D marginal prep params
- `GKP_D034_x3_to_x3y3_prep.npz` — state transfer prep params
- `fock_preparation.npz` — Fock state preparation results
- `results_vacuum.npz`, `results_Ic_improvements.npz` — floating-basis results
- `Ic_comparison_results.npz` — I_c comparison data

## Key Constants

- `GKP_N = 100` — Fock space truncation dimension
- `Delta` — GKP envelope parameter (typically 0.2–0.4)

## Notes

- All JAX code uses `jax.config.update("jax_enable_x64", True)` for double precision
- The coherent-basis pipeline (`TraceoutLayer`, `g()`, `channel_from_b()`) uses `complex64` internally for performance, while Fock-basis computations use `complex128`

### Known precision notes

- The coherent-basis pipeline (`TraceoutLayer`, `channel_from_b`, `traceout_unitary`, `compose_channel_kraus`) operates in `complex64` even when x64 is enabled. This is intentional for performance but may limit precision for very deep circuits (>15 layers) or when individual coherent terms have large displacement magnitudes.
