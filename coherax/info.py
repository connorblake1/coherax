"""Coherent information computations for bosonic codes.

Provides functions to compute the coherent information :math:`I_c` of a
bosonic code under pure-loss and thermal-loss channels from Fock-basis kets
or :class:`~coherax.states.CoherentKet` objects.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from coherax.operators import (
    make_pureloss_fock,
    make_thermalloss_fock,
    von_neumann_entropy,
)


def pureloss_coherent_info_from_kets(
    psi0: Array, psi1: Array, gamma: float, loss_rank: int = 20
) -> float:
    r"""Coherent information :math:`I_c` (qubits) from Fock-basis ket vectors.

    Computes :math:`I_c = S(\rho_B) - S(\rho_{RB})` for the maximally mixed
    logical input under a pure-loss channel with transmissivity
    :math:`\eta = 1 - \gamma`.

    Parameters
    ----------
    psi0, psi1 : Array, shape ``(N,)``
        Fock-basis ket vectors for logical 0 and 1.
    gamma : float
        Loss probability.
    loss_rank : int
        Number of Kraus operators.

    Returns
    -------
    float
        Coherent information in qubits.
    """
    N = psi0.shape[0]
    ops = make_pureloss_fock(gamma, loss_rank, N)
    v0 = jnp.einsum("knm,m->kn", ops, psi0)
    v1 = jnp.einsum("knm,m->kn", ops, psi1)
    rho_B = 0.5 * (
        jnp.einsum("kn,km->nm", v0, jnp.conj(v0))
        + jnp.einsum("kn,km->nm", v1, jnp.conj(v1))
    )
    b00 = jnp.einsum("kn,km->nm", v0, jnp.conj(v0))
    b01 = jnp.einsum("kn,km->nm", v0, jnp.conj(v1))
    b10 = jnp.einsum("kn,km->nm", v1, jnp.conj(v0))
    b11 = jnp.einsum("kn,km->nm", v1, jnp.conj(v1))
    rho_RB = 0.5 * jnp.block(
        [[b00, b01], [b10, b11]]
    )  # entropy exchange matrix Mike and Ike 12.109
    return von_neumann_entropy(rho_B) - von_neumann_entropy(rho_RB)


def pureloss_coherent_info_from_coherent_kets(
    state0: "CoherentKet",  # noqa: F821
    state1: "CoherentKet",  # noqa: F821
    gamma: float,
    loss_rank: int = 20,
) -> float:
    r"""Coherent information :math:`I_c` from :class:`CoherentKet` objects.

    Converts to Fock basis and delegates to :func:`coherent_info_from_kets`.

    Parameters
    ----------
    state0, state1 : CoherentKet
        Logical code words.
    gamma : float
        Loss probability.
    loss_rank : int
        Number of Kraus operators.

    Returns
    -------
    float
        Coherent information in qubits.
    """
    rho0 = state0.to_fock_basis()
    rho1 = state1.to_fock_basis()
    rho0 = jnp.array(rho0.to_jax() if hasattr(rho0, "to_jax") else rho0)
    rho1 = jnp.array(rho1.to_jax() if hasattr(rho1, "to_jax") else rho1)
    psi0 = jnp.linalg.eigh(rho0)[1][:, -1]
    psi1 = jnp.linalg.eigh(rho1)[1][:, -1]
    return pureloss_coherent_info_from_kets(psi0, psi1, gamma, loss_rank)


def thermalloss_coherent_info_from_kets(
    psi0: Array,
    psi1: Array,
    gamma: float,
    n_th: float,
    loss_rank: int = 20,
) -> float:
    r"""Coherent information :math:`I_c` (qubits) under thermal loss from Fock-basis kets.

    Parameters
    ----------
    psi0, psi1 : Array, shape ``(N,)``
        Fock-basis ket vectors for logical 0 and 1.
    gamma : float
        Loss probability.
    n_th : float
        Mean thermal photon number of the environment.
    loss_rank : int
        Number of Kraus operators.

    Returns
    -------
    float
        Coherent information in qubits.
    """
    N = psi0.shape[0]
    ops = make_thermalloss_fock(gamma, n_th, loss_rank, N)
    v0 = jnp.einsum("knm,m->kn", ops, psi0)
    v1 = jnp.einsum("knm,m->kn", ops, psi1)
    rho_B = 0.5 * (
        jnp.einsum("kn,km->nm", v0, jnp.conj(v0))
        + jnp.einsum("kn,km->nm", v1, jnp.conj(v1))
    )
    b00 = jnp.einsum("kn,km->nm", v0, jnp.conj(v0))
    b01 = jnp.einsum("kn,km->nm", v0, jnp.conj(v1))
    b10 = jnp.einsum("kn,km->nm", v1, jnp.conj(v0))
    b11 = jnp.einsum("kn,km->nm", v1, jnp.conj(v1))
    rho_RB = 0.5 * jnp.block([[b00, b01], [b10, b11]])
    return von_neumann_entropy(rho_B) - von_neumann_entropy(rho_RB)


# TODO use frame theory to avoid projection into Fock entirely
