"""Backward-compatibility shim.

All functionality has been moved to:
- :mod:`coherax.operators`
- :mod:`coherax.states`
- :mod:`coherax.circuits`
- :mod:`coherax.fidelity`
- :mod:`coherax.gkp`

This module re-exports everything so existing imports continue to work.
"""

# ruff: noqa: F401, F403
from coherax.operators import *
from coherax.states import *
from coherax.circuits import *
from coherax.fidelity import *
from coherax.gkp import *

# Legacy names — operators module renamed `a` → `a_op`, etc.
# Keep these aliases for notebook/script compatibility.
from coherax.operators import a_op as a, a_dag_op as a_dag, x_quad as x, p_quad as p
from coherax.circuits import _addmask as addmask, _caddmask as caddmask
