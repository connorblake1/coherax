# coherax

Coherent-basis optimization toolkit for bosonic quantum error-correcting codes.

Built on [JAX](https://github.com/jax-ml/jax) and [dynamiqs](https://github.com/dynamiqs/dynamiqs), coherax provides:

- **Analytic closed-form fidelity** via coherent-basis decomposition of CD+R circuits
- **Transpose channel recovery** for GKP codes under photon loss
- **CMA-ES worst-case optimization** ensuring robust recovery across the Bloch sphere
- **Coherent information benchmarks** comparing GKP, cat, binomial, and floating-basis codes

## Quickstart

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
```

> **Note:** scipy must be <1.14 because StrawberryFields 0.23 uses `scipy.integrate.simps`, removed in 1.14.

## Usage

```python
from coherax.characteristic_jax_utils import CoherentKet, BosonicSubspace, GKP_N
from coherax.transpose_channel_claude import build_gkp_states, entanglement_fidelity
```

Run the results notebook:

```bash
jupyter notebook aggregated_results.ipynb
```

## Project Structure

```
coherax/
├── aggregated_results.ipynb          # Main results notebook (8 sections)
├── benchmark_codes_claude.py         # Cat/binomial/GKP comparison
├── fock_fidelity_claude.py           # Fock state preparation optimization
├── testing_data/                     # Saved .npz parameters and results
├── figs/                             # Generated figures
├── coherax/                          # Core library
│   ├── characteristic_jax_utils.py   # CoherentKet, BosonicSubspace, loss channels
│   ├── transpose_channel_claude.py   # Transpose recovery, SBS baseline
│   ├── worstcase_optimizer_claude.py # CMA-ES worst-case fidelity
│   ├── coherent_tree_optimizer_claude.py
│   ├── binary_tree_utils.py
│   └── deprecated/                   # Archived optimizer/sweep scripts
└── CLAUDE.md                         # Development notes
```

## Data Files

The notebook expects `.npz` result files in `testing_data/`. See `CLAUDE.md` for the full manifest.

## Citation

C. Blake and L. Jiang, "Coherent-Basis Optimization for Bosonic Quantum Codes," 2026.

## License

TBD
