# coherax

Coherent-basis optimization toolkit for bosonic quantum error-correcting codes.

[![Documentation](https://readthedocs.org/projects/coherax/badge/?version=latest)](https://coherax.readthedocs.io)

Built on [JAX](https://github.com/jax-ml/jax) and [dynamiqs](https://github.com/dynamiqs/dynamiqs), coherax exposes two main capabilities:

- **State preparation** — gradient-based optimization of CD+R / ECD circuits for GKP, cat, Fock, and arbitrary coherent-superposition targets, using analytic closed-form fidelities that never touch the Fock basis during the inner loop.
- **Channel optimization** — joint optimization of a floating-basis coherent-state encoder and a CPTP Kraus decoder against the pure-loss channel, maximizing entanglement fidelity $F_e$ or coherent information $I_c$. Everything runs in the coherent basis with no Fock truncation in the optimizer.

A typed `Ket` / `DM` / operator hierarchy (`CoherentKet`, `FockKet`, `LogicalKet`, `JointKet`, `CoherentDM`, `FockDM`, `CoherentCoherentOp` / `FockFockOp` / `CoherentFockOp` / `FockCoherentOp`, `Displacer`, `Rotator`, `CPTP`, `BosonicSubspace`) underlies both pipelines.

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

## Quick examples

### State preparation — ECD circuit for a 2-legged cat

```python
import jax.numpy as jnp
import dynamiqs as dq
from coherax import CoherentKet, optimize_ECD_state_prep, state_fidelity, CircuitUnitary, QubitKet

target = CoherentKet(cs=jnp.array([1.0, 1.0]), ds=jnp.array([4.0+0j, -4.0+0j]))
params, infid = optimize_ECD_state_prep(target_state=target, N_depth=6, restarts=2, steps=15000, lr=1e-3)

# Verify with the typed circuit API
U = CircuitUnitary.from_params(params, N_l=2**6)
vac = CoherentKet(cs=jnp.array([1.0]), ds=jnp.array([0.0+0j]))
q0 = QubitKet(cs=jnp.array([1.0, 0.0]))
output = U.apply(vac, q0).inner(q0)
print(f"F = {float(state_fidelity(target, output)):.6f}")
dq.plot.wigner(output.to_fock_basis())
```

See `demo.ipynb` for the full set of state-prep results (GKP pipelines, X3/X4 1D marginals, Fock states |1⟩–|8⟩ at depths 4–10, cat-to-cat transfer).

### Channel optimization — floating-basis encoder/decoder under pure loss

See `floating_basis.ipynb` for a runnable verification notebook with plots (F_e/I_c vs γ, convergence curves, phase-space scatter, Wigner functions).

![GKP state preparation](testing_data/gkp_prep.gif)

## Project structure

```
coherax/
├── demo.ipynb                       # State preparation walkthrough
├── floating_basis.ipynb             # Channel optimization walkthrough
├── testing_data/                    # Saved .npz parameters / reference data
├── docs/                            # Sphinx documentation
├── coherax/                         # Core library
│   ├── linalg_utils.py              # GKP_N, coherent-state kernels (aOmegab, coherent_overlap),
│   │                                # sparse eigh, support-restricted invsqrtm, complex_normal
│   ├── _fock.py                     # Dynamiqs glue + pre-built Fock-basis constants
│   │                                # (IN, sigma_x/y/z, a_op, ...) and Kraus-channel
│   │                                # utilities (apply_kraus_map, make_pureloss_fock, ...).
│   │                                # Transitional — the dynamiqs wrappers will be removed.
│   ├── states.py                    # Ket/DM hierarchy, typed basis-defined operators
│   │                                # (CoherentCoherentOp / FockFockOp / CoherentFockOp /
│   │                                # FockCoherentOp), analytic operators (Displacer,
│   │                                # Rotator, CPTP), BosonicSubspace, beamsplit_full,
│   │                                # unitary_encoding_map (floating-basis encoder).
│   ├── circuits.py                  # CD/ECD/rotation unitaries, TraceoutLayer, g(),
│   │                                # CircuitUnitary, channel_from_b.
│   ├── fidelity.py                  # Analytic CD+R fidelity, state_fidelity, circuit_*_fidelity,
│   │                                # entanglement_fidelity_pureloss,
│   │                                # coherent_information_pureloss, nbar_logical.
│   ├── gkp.py                       # GKP codeword generators (square / hex / rectangular).
│   └── optimizers.py                # optimize_ECD_state_prep / _state_transfer,
│                                    # optimize_Fe_floating / _Ic_floating,
│                                    # init_separated_d, separation_penalty.
└── pyproject.toml
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

Run a specific test file:

```bash
pytest tests/test_states.py -v
pytest tests/test_floating_basis.py -v
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
