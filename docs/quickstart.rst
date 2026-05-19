Quickstart
==========

Installation
------------

.. code-block:: bash

   pip install coherax

Or from source:

.. code-block:: bash

   git clone https://github.com/connorblake1/coherax.git
   cd coherax
   pip install -e ".[dev]"

Basic Usage
-----------

Construct a GKP code word:

.. code-block:: python

   from coherax import gkp_coherent_dm, GKP_N
   import jax.numpy as jnp

   # Logical |0> with Delta=0.3, square lattice, 3-term truncation
   log0 = gkp_coherent_dm(mu=0, N_trunc=3, Delta=0.3, lattice="square")

Compute analytic fidelity between a circuit output and a target state:

.. code-block:: python

   from coherax import g, analytic_fidelity_wrapper

   # Assume circuit_params is a (n_layers, 4) array
   N_l = 2 ** n_layers
   fidelity = analytic_fidelity_wrapper(log0, circuit_params, N_l)

Construct and apply a pure-loss channel:

.. code-block:: python

   from coherax import make_pureloss_fock, apply_kraus_map

   gamma = 0.05  # 5% loss
   loss_ops = make_pureloss_fock(gamma, rank=10)
   rho_out = apply_kraus_map(loss_ops, log0.to_fock_basis())
