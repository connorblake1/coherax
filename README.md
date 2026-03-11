# coherax

Coherent-basis optimization toolkit for bosonic quantum error-correcting codes.

[![Documentation](https://readthedocs.org/projects/coherax/badge/?version=latest)](https://coherax.readthedocs.io)

Built on [JAX](https://github.com/jax-ml/jax) and [dynamiqs](https://github.com/dynamiqs/dynamiqs), coherax provides:

- **Analytic closed-form fidelity** via coherent-basis decomposition of CD+R circuits
- **Transpose channel recovery** for GKP codes under photon loss
- **CMA-ES worst-case optimization** ensuring robust recovery across the Bloch sphere
- **Coherent information benchmarks** comparing GKP, cat, binomial, and floating-basis codes

## Installation

```bash
pip install -e ".[dev]"
```

Or with conda:

```bash
conda create -n coherax python=3.11 -y
conda activate coherax
pip install -e ".[dev]"
```

> **Note:** scipy must be <1.14 because StrawberryFields 0.23 uses `scipy.integrate.simps`, removed in 1.14.

## Usage

```python
from coherax import CoherentKet, gkp_coherent_dm, GKP_N
from coherax import g, analytic_fidelity_wrapper
from coherax import make_pureloss_fock, apply_kraus_map

# Construct GKP code words
log0 = gkp_coherent_dm(mu=0, N_trunc=3, Delta=0.3, lattice="square")
log1 = gkp_coherent_dm(mu=1, N_trunc=3, Delta=0.3, lattice="square")

# Apply pure loss and recover
loss_ops = make_pureloss_fock(gamma=0.05, rank=10)
rho_out = apply_kraus_map(loss_ops, log0.to_fock_basis())
```

Run the results notebook:

```bash
jupyter notebook aggregated_results.ipynb
```

## Project Structure

```
coherax/
├── aggregated_results.ipynb          # Main results notebook
├── benchmark_codes_claude.py         # Cat/binomial/GKP comparison
├── fock_fidelity_claude.py           # Fock state preparation
├── testing_data/                     # Saved .npz parameters
├── figs/                             # Generated figures
├── docs/                             # Sphinx documentation
├── coherax/                          # Core library
│   ├── operators.py                  # Quantum operators, constants, channels
│   ├── states.py                     # CoherentKet, CoherentDM, BosonicSubspace
│   ├── circuits.py                   # CD+R circuits, TraceoutLayer, g()
│   ├── fidelity.py                   # Analytic fidelity computations
│   ├── gkp.py                        # GKP code state generators, diagnostics
│   ├── info.py                       # Coherent information computations
│   ├── transpose_channel_claude.py   # Transpose recovery
│   └── worstcase_optimizer_claude.py # CMA-ES optimization
└── pyproject.toml
```

## Documentation

Full API docs: [coherax.readthedocs.io](https://coherax.readthedocs.io)

## Citation

C. Blake and L. Jiang, "Coherent-Basis Optimization for Bosonic Quantum Codes," 2026.

## License

MIT
