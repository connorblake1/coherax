# coherax

Coherent-basis optimization toolkit for bosonic quantum error-correcting codes.

[![Documentation](https://readthedocs.org/projects/coherax/badge/?version=latest)](https://coherax.readthedocs.io)

Built on [JAX](https://github.com/jax-ml/jax) and [dynamiqs](https://github.com/dynamiqs/dynamiqs), coherax provides:

- **Analytic closed-form fidelity** via coherent-basis decomposition of CD+R circuits
- **GKP, cat, and Fock state preparation** with gradient-based ECD optimization
- **Coherent information benchmarks** comparing GKP, cat, binomial, and floating-basis codes
- **Transpose channel recovery** for bosonic codes under photon loss

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

See `demo.ipynb` for worked examples including state preparation, Fock state optimization, and coherent information benchmarks.

## Project Structure

```
coherax/
├── demo.ipynb                       # Main demo notebook
├── testing_data/                    # Saved .npz parameters
├── figs/                            # Generated figures
├── docs/                            # Sphinx documentation
├── coherax/                         # Core library
│   ├── operators.py                 # Quantum operators, constants, channels
│   ├── states.py                    # CoherentKet, CoherentDM, BosonicSubspace
│   ├── circuits.py                  # CD+R circuits, TraceoutLayer, g()
│   ├── fidelity.py                  # Analytic fidelity computations
│   ├── gkp.py                       # GKP code state generators
│   ├── info.py                      # Coherent information computations
│   └── optimizers.py                # ECD circuit optimization
└── pyproject.toml
```

## Documentation

Full API docs: [coherax.readthedocs.io](https://coherax.readthedocs.io)

## Citation

C. Blake, G. Zheng, G. Lee, and L. Jiang, "Coherent-Basis Optimization for Bosonic Quantum Codes," 2026.

## License

MIT
