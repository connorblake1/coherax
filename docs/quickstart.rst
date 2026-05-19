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

Hilbert-space conventions
-------------------------

* Joint cavity--qubit states use the ``dqtensor(cavity, qubit) =
  kron(cavity, qubit)`` ordering (cavity index slow, qubit index fast).
  This is the layout produced by every ``circuits.py`` unitary
  (:func:`coherax.circuits.CD`, :func:`coherax.circuits.ECD`,
  :func:`coherax.circuits.circuit_layer`) and by
  :meth:`coherax.states.JointKet.to_fock_ket`.
* The Fock-space truncation constant :data:`coherax.GKP_N` defaults to
  100. Override per-call via the ``N`` keyword on every state's
  ``to_fock_ket(N)`` / ``to_fock_basis(N)``.
* JAX must run with ``x64`` enabled. Add this once at the top of any
  script or notebook:

  .. code-block:: python

     import jax
     jax.config.update("jax_enable_x64", True)

Typed state hierarchy
---------------------

The library exposes a typed :class:`~coherax.states.Ket` /
:class:`~coherax.states.DM` hierarchy with closed-form inner products
and lazy Fock-basis conversion:

.. code-block:: python

   import jax.numpy as jnp
   from coherax import CoherentKet, FockKet, LogicalKet, QubitKet, state_fidelity

   cat = CoherentKet(cs=jnp.array([1.0, 1.0]),
                     ds=jnp.array([2.0+0j, -2.0+0j]))
   fock3 = FockKet(cs=jnp.array([1.0]), ns=jnp.array([3]))
   print(state_fidelity(cat, fock3))             # |<cat|3>|^2

   # D-dim orthonormal logical subspace via LogicalKet
   logical = LogicalKet(cs=jnp.array([0.6, 0.8j]),
                        ns=jnp.array([0, 2]))    # span {|0>, |2>}

   # Typed two-level shorthand
   qubit = QubitKet(cs=jnp.array([1.0, 0.0]))

Basis-defined operators are split by domain / codomain basis:
:class:`~coherax.states.CoherentCoherentOp`,
:class:`~coherax.states.FockFockOp`,
:class:`~coherax.states.CoherentFockOp`,
:class:`~coherax.states.FockCoherentOp`. Each stores its basis kets as
stacked ``(M, A)`` arrays for ``jax.jit``-friendly application and
exposes ``apply``, ``apply_adj``, ``dagger``, and ``wrap(rho)``.

State preparation: ECD circuits
-------------------------------

Use closed-form fidelity formulas to optimize a CD+R / ECD circuit that
prepares a target coherent superposition or Fock state.

.. code-block:: python

   import jax.numpy as jnp
   from coherax import (
       CoherentKet, CircuitUnitary, QubitKet,
       optimize_ECD_state_prep, state_fidelity,
   )

   target = CoherentKet(cs=jnp.array([1.0, 1.0]),
                        ds=jnp.array([4.0+0j, -4.0+0j]))
   params, infid = optimize_ECD_state_prep(
       target_state=target, N_depth=6, restarts=2, steps=15000, lr=1e-3,
   )

   # Verify with the typed circuit API
   U = CircuitUnitary.from_params(params, N_l=2**6)
   vac = CoherentKet(cs=jnp.array([1.0]), ds=jnp.array([0.0+0j]))
   q0 = QubitKet(cs=jnp.array([1.0, 0.0]))
   output = U.apply(vac, q0).inner(q0)
   print(f"F = {float(state_fidelity(target, output)):.6f}")

For a Fock-state target with fidelity
:math:`F_m = \sum_j |\sum_i \alpha_{ji}\langle m|\beta_{ji}\rangle|^2`,
use :func:`coherax.fidelity.circuit_fock_fidelity` (or the
:func:`coherax.fidelity.analytic_fidelity_fock_wrapper` lower-level
analytic form).

Construct a GKP code word for use as a target:

.. code-block:: python

   from coherax import gkp_coherent_dm, GKP_N

   # Logical |0> with Delta=0.34, square lattice, 3-term truncation
   gkp_x3 = gkp_coherent_dm(mu=0, N_trunc=3, Delta=0.34, lattice="square", N_trunc_y=0)

(Note: despite its name, :func:`~coherax.gkp.gkp_coherent_dm` returns a
:class:`~coherax.states.CoherentKet`.)

Pure-loss channels (Fock-basis)
-------------------------------

When you need explicit Kraus operators in the Fock basis (e.g. for
trace-distance computations or non-coherent inputs):

.. code-block:: python

   from coherax import make_pureloss_fock, apply_kraus_map

   gamma = 0.05
   loss_ops = make_pureloss_fock(gamma, rank=10)
   rho_out = apply_kraus_map(loss_ops, gkp_x3.to_fock_basis())

Channel optimization: floating basis
------------------------------------

The floating-basis encoder/decoder pipeline maximizes entanglement
fidelity :math:`F_e` (joint encoder + Kraus decoder) or coherent
information :math:`I_c` (encoder-only, decoder-independent) of a code
under pure photon loss. All math stays in the coherent basis -- no
Fock truncation in the inner optimization loop.

.. code-block:: python

   from coherax import optimize_Fe_floating, optimize_Ic_floating

   # Joint encoder + CPTP decoder maximizing F_e at gamma=0.10
   res_fe = optimize_Fe_floating(
       gamma=0.10, N_C=10, N_D=10, restarts=5,
       steps_p1=2000, steps_p2=1000,
   )
   print(f"F_e = {res_fe['Fe']:.6f}, nbar = {res_fe['nbar']:.2f}")
   # res_fe['X'], res_fe['d'], res_fe['Z']  -- best encoder + decoder params

   # Encoder-only I_c optimization (decoder is irrelevant for I_c)
   res_ic = optimize_Ic_floating(
       gamma=0.10, N_C=10, restarts=5,
       steps_p1=2000, steps_p2=1000,
   )
   print(f"I_c = {res_ic['Ic']:.6f} qubits, nbar = {res_ic['nbar']:.2f}")

The encoder is parametrized by an unconstrained matrix
:math:`X \in \mathbb{C}^{N_C \times D}` and a displacement vector
:math:`d \in \mathbb{C}^{N_C}`; the algebraic isometry
:math:`C = G^{-1/2}|_{\mathrm{supp}}\, X\, (X^\dagger X)^{-1/2}|_{\mathrm{supp}}`
guarantees :math:`C^\dagger G C = I_D` so no constraint enforcement is
required. See :func:`coherax.states.unitary_encoding_map` for the raw
kernel and :func:`coherax.states.encode_logical_kets` for the typed
:class:`~coherax.states.CoherentKet` entry point.

Compute :math:`F_e` and :math:`I_c` directly from saved encoder /
decoder parameters:

.. code-block:: python

   from coherax import (
       entanglement_fidelity_pureloss,
       coherent_information_pureloss,
       nbar_logical,
   )

   Fe = float(entanglement_fidelity_pureloss(res_fe['X'], res_fe['d'], res_fe['Z'], gamma=0.10))
   Ic = float(coherent_information_pureloss(res_ic['X'], res_ic['d'], gamma=0.10))
   nbar = float(nbar_logical(res_fe['X'], res_fe['d']))

The same beamsplitter kernel that drives the optimization is also
available as a user-facing function,
:func:`coherax.states.beamsplit_full`, which accepts a list of typed
:class:`~coherax.states.CoherentKet` logical states sharing a common
displacement vector.

Worked examples
---------------

* ``demo.ipynb`` -- full state-preparation walkthrough (GKP, Fock,
  cat-to-cat transfer).
* ``floating_basis.ipynb`` -- channel-optimization walkthrough with
  F_e/I_c sweeps over loss rates, convergence curves, phase-space
  scatter, and Wigner functions of the optimized logical states.
