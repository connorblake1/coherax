coherax
=======

Coherent-basis optimization toolkit for bosonic quantum error-correcting
codes.

Built on `JAX <https://github.com/jax-ml/jax>`_ and
`dynamiqs <https://github.com/dynamiqs/dynamiqs>`_, coherax exposes
two main capabilities:

* **State preparation.** Gradient-based optimization of CD+R / ECD
  circuits for GKP, cat, Fock, and arbitrary coherent-superposition
  targets, using analytic closed-form fidelities that never touch the
  Fock basis during the inner loop.
* **Channel optimization.** Joint optimization of a floating-basis
  coherent-state encoder and a CPTP Kraus decoder against the pure-loss
  channel, maximizing entanglement fidelity :math:`F_e` or coherent
  information :math:`I_c`. Everything runs in the coherent basis with
  no Fock truncation in the optimizer.

A typed :class:`~coherax.states.Ket` / :class:`~coherax.states.DM` /
operator hierarchy (:class:`~coherax.states.CoherentKet`,
:class:`~coherax.states.FockKet`, :class:`~coherax.states.LogicalKet`,
:class:`~coherax.states.JointKet`, :class:`~coherax.states.CoherentDM`,
:class:`~coherax.states.FockDM`,
:class:`~coherax.states.CoherentCoherentOp` and three sibling typed
operators, plus :class:`~coherax.states.Displacer`,
:class:`~coherax.states.Rotator`, :class:`~coherax.states.CPTP`, and
:class:`~coherax.states.BosonicSubspace`) underlies both pipelines.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   api/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
