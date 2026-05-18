# coherax

Coherent-basis optimization toolkit for bosonic quantum error-correcting codes.

[![Documentation](https://readthedocs.org/projects/coherax/badge/?version=latest)](https://coherax.readthedocs.io)

Built on [JAX](https://github.com/jax-ml/jax) and [dynamiqs](https://github.com/dynamiqs/dynamiqs), coherax provides:

- **Analytic closed-form fidelity** via coherent-basis decomposition of CD+R circuits
- **GKP, cat, and Fock state preparation** with gradient-based ECD optimization
- **Coherent information benchmarks** comparing GKP, cat, binomial, and floating-basis codes
- More to come!

> **Warning:** This library is under active development. The API is unstable and may change without notice between releases.

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

See `demo.ipynb` for worked examples including state preparation, Fock state optimization, and coherent information benchmarks. For example, here is a 23 ECD preparation of a high fidelity (.9998) finite GKP state:

![GKP state preparation](testing_data/gkp_prep.gif)

## Project Structure

```
coherax/
├── demo.ipynb                       # Main demo notebook
├── testing_data/                    # Saved .npz parameters
├── docs/                            # Sphinx documentation
├── coherax/                         # Core library
│   ├── linalg_utils.py              # GKP_N, coherent-state kernels, sparse eigh
│   ├── _fock.py                     # Dynamiqs glue, Fock-basis constants, Kraus channels (transitional)
│   ├── states.py                    # Ket/DM hierarchy, typed operators, CPTP
│   ├── circuits.py                  # CD+R circuits, TraceoutLayer, g()
│   ├── fidelity.py                  # Analytic fidelity computations
│   ├── gkp.py                       # GKP code state generators
│   └── optimizers.py                # ECD circuit optimization
└── pyproject.toml
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

Or run a specific test file:

```bash
pytest tests/test_states.py -v
```

## Documentation

Full API docs: [coherax.readthedocs.io](https://coherax.readthedocs.io)

## Citation

If you use coherax in your research, please cite:

```bibtex
@software{coherax2026,
  author       = {Blake, Connor and Zheng, Guo and Lee, Gideon and Jiang, Liang},
  title        = {coherax: Coherent-basis optimization for bosonic quantum codes},
  year         = {2026},
  url          = {https://github.com/connorblake1/coherax},
  version      = {0.1.0},
}
```

## License

MIT
