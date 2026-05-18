"""Dynamiqs glue, pre-built Fock-basis constants, and Fock-basis channel utilities.

This module is transitional: the dynamiqs wrappers will be removed once the
benchmarking paths against ``dq`` are no longer needed. Kept private (leading
underscore) to discourage external imports.

Contents:

- Thin JAX-compatible wrappers around dynamiqs (``dqtensor``, ``dqcoherent``,
  ``dqdisplace``, ``dqdag``, ...).
- Pre-built constants at :data:`coherax.linalg_utils.GKP_N`: identities,
  Pauli operators, bosonic ladder operators, quadratures, qubit basis kets.
- Fock-basis Kraus-map utilities: :func:`apply_kraus_map`,
  :func:`compose_channel_kraus`, :func:`make_pureloss_fock`,
  :func:`make_thermalloss_fock`, :func:`make_transpose_for_pureloss`,
  :func:`von_neumann_entropy`.
"""

from __future__ import annotations

import math
from functools import partial

import dynamiqs as dq
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np
import scipy.linalg as sla
from jaxtyping import Array

from coherax.linalg_utils import GKP_N, dag

# ---------------------------------------------------------------------------
# dynamiqs -> JAX wrappers
# ---------------------------------------------------------------------------


@jax.jit
def dqtensor(*args: Array) -> Array:
    """Tensor product of operators via dynamiqs, returned as a JAX array."""
    return dq.tensor(*args).to_jax()


@jax.jit
def dqdag(arg: Array) -> Array:
    """Conjugate transpose via dynamiqs, returned as a JAX array."""
    return dq.dag(arg).to_jax()


def dqeye(n: int) -> Array:
    """Identity matrix of dimension *n*."""
    return dq.eye(n).to_jax()


def dqnumber(n: int) -> Array:
    """Number operator :math:`\\hat{n}` truncated to *n* levels."""
    return dq.number(n).to_jax()


def dqdestroy(n: int) -> Array:
    """Lowering (annihilation) operator truncated to *n* levels."""
    return dq.destroy(n).to_jax()


def dqcreate(n: int) -> Array:
    """Raising (creation) operator truncated to *n* levels."""
    return dq.create(n).to_jax()


@jax.jit
def dqtrace(arg: Array) -> Array:
    """Trace of an operator."""
    return dq.trace(arg)


@partial(jax.jit, static_argnums=0)
def dqdisplace(n: int, alpha: complex) -> Array:
    """Displacement operator :math:`D(\\alpha)` truncated to *n* levels."""
    return dq.displace(n, alpha).to_jax()


@partial(jax.jit, static_argnums=0)
def dqsqueeze(n: int, z: complex) -> Array:
    """Squeeze operator :math:`S(z)` truncated to *n* levels."""
    return dq.squeeze(n, z).to_jax()


@partial(jax.jit, static_argnums=0)
def dqfock_dm(n: int, k: int) -> Array:
    """Fock-state density matrix :math:`|k\\rangle\\langle k|`."""
    return dq.fock_dm(n, k).to_jax()


@partial(jax.jit, static_argnums=0)
def dqcoherent_dm(n: int, alpha: complex) -> Array:
    """Coherent-state density matrix :math:`|\\alpha\\rangle\\langle\\alpha|`."""
    return dq.coherent_dm(n, alpha).to_jax()


@partial(jax.jit, static_argnums=0)
def dqcoherent(n: int, alpha: complex) -> Array:
    """Coherent-state ket :math:`|\\alpha\\rangle`."""
    return dq.coherent(n, alpha).to_jax()


@partial(jax.jit, static_argnums=(1, 2))
def dqptrace(rho: Array, keep: int, dims: tuple[int, ...]) -> Array:
    """Partial trace, keeping subsystem *keep*."""
    return dq.ptrace(rho, keep, dims).to_jax()


@jax.jit
def dqexpect(O: Array, rho: Array) -> Array:
    """Expectation value :math:`\\mathrm{Tr}[O \\rho]`."""
    return dq.expect(O, rho)


@jax.jit
def dqtodm(psi: Array) -> Array:
    """Convert a ket to a density matrix :math:`|\\psi\\rangle\\langle\\psi|`."""
    return dq.todm(psi).to_jax()


# ---------------------------------------------------------------------------
# Pre-built operators at GKP_N
# ---------------------------------------------------------------------------

root2: Array = jnp.sqrt(2.0)
"""Square root of 2."""

IN: Array = dqeye(GKP_N)
"""Identity on the bosonic mode (GKP_N x GKP_N)."""

I2: Array = dqeye(2)
"""Identity on the qubit (2 x 2)."""

sigma_x: Array = dq.sigmax().to_jax()
"""Pauli X."""

sigma_y: Array = dq.sigmay().to_jax()
"""Pauli Y."""

sigma_z: Array = dq.sigmaz().to_jax()
"""Pauli Z."""

n_hat: Array = dqnumber(GKP_N)
"""Number operator :math:`\\hat{n}` at truncation ``GKP_N``."""

a_op: Array = dqdestroy(GKP_N)
"""Annihilation operator at truncation ``GKP_N``."""

a_dag_op: Array = dqcreate(GKP_N)
"""Creation operator at truncation ``GKP_N``."""

x_quad: Array = (a_op + a_dag_op) / root2
r"""Position quadrature :math:`\hat{x} = (\hat{a} + \hat{a}^\dagger)/\sqrt{2}`."""

p_quad: Array = -1.0j * (a_op - a_dag_op) / root2
r"""Momentum quadrature :math:`\hat{p} = -i(\hat{a} - \hat{a}^\dagger)/\sqrt{2}`."""

ket0: Array = dq.fock(2, 0)
"""Qubit ground state :math:`|0\\rangle`."""

ket1: Array = dq.fock(2, 1)
"""Qubit excited state :math:`|1\\rangle`."""


# ---------------------------------------------------------------------------
# Quantum channels (Fock-basis Kraus representation)
# ---------------------------------------------------------------------------


@jax.jit
def apply_kraus_map_nonorm(ops: Array, rho: Array) -> Array:
    r"""Apply a Kraus map :math:`\sum_k K_k \rho K_k^\dagger` without normalizing.

    Parameters
    ----------
    ops : Array, shape ``(K, N, N)``
        Kraus operators.
    rho : Array, shape ``(N, N)``
        Input density matrix.

    Returns
    -------
    Array, shape ``(N, N)``
        Output density matrix (un-normalized).
    """
    return jnp.sum(jax.vmap(lambda op: op @ rho @ dqdag(op))(ops), axis=0)


@jax.jit
def apply_kraus_map(ops: Array, rho: Array) -> Array:
    r"""Apply a Kraus map and trace-normalize the output.

    Parameters
    ----------
    ops : Array, shape ``(K, N, N)``
        Kraus operators.
    rho : Array, shape ``(N, N)``
        Input density matrix.

    Returns
    -------
    Array, shape ``(N, N)``
        Trace-normalized output density matrix.
    """
    rho_out = apply_kraus_map_nonorm(ops, rho)
    return rho_out / dqtrace(rho_out)


@jax.jit
def apply_kraus_map_n(ops: Array, rho: Array, n: int) -> Array:
    """Apply a Kraus map *n* times in sequence.

    Parameters
    ----------
    ops : Array, shape ``(K, N, N)``
        Kraus operators.
    rho : Array, shape ``(N, N)``
        Input density matrix.
    n : int
        Number of applications.

    Returns
    -------
    Array, shape ``(N, N)``
        Trace-normalized output after *n* rounds.
    """

    def body_loop(i: int, rho_loop: Array) -> Array:
        return apply_kraus_map(ops, rho_loop)

    rho_out = jax.lax.fori_loop(0, n, body_loop, rho)
    return rho_out / dqtrace(rho_out)


@jax.jit
def compose_channel_kraus(ch1: Array, ch2: Array) -> Array:
    """Compose two quantum channels in Kraus representation.

    Returns Kraus operators for the channel ``ch1 . ch2`` (ch2 applied first).

    Parameters
    ----------
    ch1 : Array, shape ``(K1, N, N)``
    ch2 : Array, shape ``(K2, N, N)``

    Returns
    -------
    Array, shape ``(K1*K2, N, N)``
    """
    new_size = ch1.shape[0] * ch2.shape[0]
    new_ops = jnp.zeros((new_size, ch1.shape[1], ch2.shape[2]), dtype=jnp.complex64)
    for i in range(ch1.shape[0]):
        for j in range(ch2.shape[0]):
            new_ops = new_ops.at[i * ch2.shape[0] + j, :, :].set(
                ch1[i, :, :] @ ch2[j, :, :]
            )
    return new_ops


def make_pureloss_fock(gamma: float, rank: int, N: int = GKP_N) -> Array:
    r"""Kraus operators for the pure-loss (amplitude damping) channel.

    .. math::
        K_l = \sqrt{\binom{}{} \frac{\gamma}{1-\gamma}}^{l/2}
              \frac{\hat{a}^l}{\sqrt{l!}} e^{\frac{\ln(1-\gamma)}{2}\hat{n}}

    Parameters
    ----------
    gamma : float
        Loss probability in ``[0, 1)``.
    rank : int
        Number of Kraus operators (photon-loss truncation).
    N : int
        Hilbert space dimension.

    Returns
    -------
    Array, shape ``(rank, N, N)``
    """
    n_op = dqnumber(N)
    a_hat = dqdestroy(N)
    return jnp.array(
        [
            (gamma / (1 - gamma)) ** (l / 2)
            / jnp.sqrt(math.factorial(l))
            * jnp.linalg.matrix_power(a_hat, l)
            @ jla.expm(jnp.log(1 - gamma) * n_op / 2)
            for l in range(rank)
        ]
    )


def make_transpose_for_pureloss(
    loss_ops: Array,
    logical_0: "CoherentKet",  # noqa: F821  (forward ref to avoid circular import)
    logical_1: "CoherentKet",  # noqa: F821
    eps: float = 1e-5,
) -> Array:
    r"""Petz transpose (near-optimal) recovery channel for pure loss.

    Constructs :math:`\mathcal{R}^T` from the code projector
    :math:`P_C = |0_L\rangle\langle 0_L| + |1_L\rangle\langle 1_L|`
    and the loss channel Kraus operators.

    Parameters
    ----------
    loss_ops : Array, shape ``(K, N, N)``
        Kraus operators of the loss channel.
    logical_0, logical_1 : CoherentKet
        Logical code words (anything with a ``to_fock_basis()`` method works).
    eps : float
        Eigenvalue cutoff for pseudo-inverse.

    Returns
    -------
    Array, shape ``(K, N, N)``
        Recovery Kraus operators.
    """
    P = logical_0.to_fock_basis() + logical_1.to_fock_basis()
    loss_P = apply_kraus_map_nonorm(loss_ops, P)
    loss_P_eigs, loss_P_vecs = jnp.linalg.eigh(loss_P)

    def supp_invsqrt(arr: Array) -> Array:
        return jnp.where(arr != 0, arr**-0.5, arr)

    loss_P_eigs2 = supp_invsqrt(jnp.round(loss_P_eigs, decimals=int(-jnp.log10(eps))))
    loss_P_invsqrt = loss_P_vecs @ jnp.diag(loss_P_eigs2) @ dqdag(loss_P_vecs)
    inv_loss_ops = jnp.array(
        [dqdag(loss_ops[i, :, :]) for i in range(loss_ops.shape[0])]
    )
    return jnp.array(
        [P @ inv_loss_ops[i, :, :] @ loss_P_invsqrt for i in range(loss_ops.shape[0])]
    )


def von_neumann_entropy(rho: Array) -> float:
    r"""Von Neumann entropy :math:`S(\rho) = -\mathrm{Tr}(\rho \log_2 \rho)` in qubits.

    Parameters
    ----------
    rho : Array, shape ``(N, N)``
        Density matrix.

    Returns
    -------
    float
        Entropy in qubits.
    """
    evals = jnp.linalg.eigvalsh(rho)
    evals = jnp.real(evals)
    return float(-jnp.sum(jnp.where(evals > 1e-15, evals * jnp.log2(evals), 0.0)))


def make_thermalloss_fock(
    gamma: float, n_th: float, rank: int = 20, N: int = GKP_N
) -> Array:
    r"""Kraus operators for the thermal-loss channel.

    Constructs the channel via a beam-splitter interaction with a thermal
    environment and partial trace over environment modes.

    Parameters
    ----------
    gamma : float
        Loss probability in ``[0, 1)``.
    n_th : float
        Mean thermal photon number of the environment.
    rank : int
        Number of Kraus operators (environment truncation).
    N : int
        Hilbert space dimension.

    Returns
    -------
    Array, shape ``(rank, N, N)``
    """
    if n_th < 1e-12:
        return make_pureloss_fock(gamma, rank, N)
    eta = 1.0 - gamma
    a = np.array(dqdestroy(N))
    N_env_trunc = min(rank, 15)
    p_thermal = np.array(
        [(n_th**k / (1 + n_th) ** (k + 1)) for k in range(N_env_trunc)]
    )
    p_thermal = p_thermal / p_thermal.sum()
    a_sys = np.kron(a, np.eye(N_env_trunc))
    a_env = np.kron(np.eye(N), np.array(dqdestroy(N_env_trunc)))
    theta = np.arccos(np.sqrt(eta))
    H_BS = theta * (a_sys.conj().T @ a_env - a_sys @ a_env.conj().T)
    U_BS = sla.expm(H_BS)
    kraus_ops = []
    for j in range(N_env_trunc):
        K_j = np.zeros((N, N), dtype=np.complex128)
        for k in range(N_env_trunc):
            block = U_BS[np.arange(N) * N_env_trunc + j][
                :, np.arange(N) * N_env_trunc + k
            ]
            K_j += np.sqrt(p_thermal[k]) * block
        kraus_ops.append(K_j)
    return jnp.array(np.array(kraus_ops))
