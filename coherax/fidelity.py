"""Analytic fidelity computations in the coherent basis.

All functions operate on the ``(alpha, beta)`` representation produced by
:func:`coherax.circuits.g` and compute fidelities without Fock-space
simulation.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from jaxtyping import Array

from coherax.linalg_utils import aOmegab


# ---------------------------------------------------------------------------
# Core fidelity kernels
# ---------------------------------------------------------------------------


@jax.jit
def analytic_fidelity_i(
    coeffs_a: Array,
    coeffs_b: Array,
    peaks_a: Array,
    peaks_b: Array,
) -> Array:
    r"""Fidelity :math:`|\langle\psi_a|\psi_b\rangle|^2` between two coherent superpositions.

    Parameters
    ----------
    coeffs_a : Array, shape ``(A,)``
        Coefficients of state *a*.
    coeffs_b : Array, shape ``(B,)``
        Coefficients of state *b*.
    peaks_a : Array, shape ``(A,)``
        Displacement amplitudes of state *a*.
    peaks_b : Array, shape ``(B,)``
        Displacement amplitudes of state *b*.

    Returns
    -------
    Array
        Scalar fidelity.
    """
    A = peaks_a.shape[0]
    B = peaks_b.shape[0]
    ca = coeffs_a.reshape(A, 1)
    da = peaks_a.reshape(A, 1)
    alphaj = coeffs_b.reshape(1, B)
    betaj = peaks_b.reshape(1, B)
    prefactor = jnp.conj(ca) * alphaj
    envelope = jnp.exp(-0.5 * jnp.abs(betaj - da) ** 2)
    phase = jnp.exp(1j * aOmegab(da, betaj))
    return jnp.abs(jnp.sum(prefactor * envelope * phase)) ** 2


@jax.jit
def analytic_fidelity(
    all_coeffs_a: Array,
    all_coeffs_b: Array,
    all_peaks_a: Array,
    all_peaks_b: Array,
) -> Array:
    r"""Batched fidelity summing over Kraus branches.

    Computes :math:`\sum_{i,j} F_i(a_i, b_j)` where each term
    is an :func:`analytic_fidelity_i` call.

    Parameters
    ----------
    all_coeffs_a : Array, shape ``(N, A)``
    all_coeffs_b : Array, shape ``(M, B)``
    all_peaks_a : Array, shape ``(N, A)``
    all_peaks_b : Array, shape ``(M, B)``

    Returns
    -------
    Array
        Scalar fidelity.
    """
    N = all_peaks_a.shape[0]
    M = all_peaks_b.shape[0]

    def body_i(i: int, acci: Array) -> Array:
        def body_j(j: int, accj: Array) -> Array:
            return accj + analytic_fidelity_i(
                all_coeffs_a[i], all_coeffs_b[j], all_peaks_a[i], all_peaks_b[j]
            )

        return jax.lax.fori_loop(0, M, body_j, acci)

    return jax.lax.fori_loop(0, N, body_i, 0.0)


# ---------------------------------------------------------------------------
# State-transfer fidelity
# ---------------------------------------------------------------------------


@jax.jit
def analytic_fidelity_transfer_i(
    alpha_i: Array,
    beta_i: Array,
    c: Array,
    d: Array,
    cp: Array,
    dp: Array,
) -> Array:
    r"""Fidelity for a single Kraus branch of a state-transfer channel.

    Measures how well a circuit maps an initial coherent superposition
    ``(c, d)`` to a target ``(cp, dp)``.

    Parameters
    ----------
    alpha_i, beta_i : Array, shape ``(N_l,)``
        Channel coefficients and displacements for one Kraus branch.
    c, d : Array
        Initial-state coefficients and displacements.
    cp, dp : Array
        Target-state coefficients and displacements.

    Returns
    -------
    Array
        Scalar fidelity contribution.
    """
    N = alpha_i.shape[0]
    A = c.shape[0]
    Ap = cp.shape[0]
    alpha_i = alpha_i.reshape((N, 1, 1))
    beta_i = beta_i.reshape((N, 1, 1))
    c = c.reshape((1, A, 1))
    d = d.reshape((1, A, 1))
    cp = cp.reshape((1, 1, Ap))
    dp = dp.reshape((1, 1, Ap))
    prefactor = alpha_i * c * jnp.conj(cp)
    exponential = jnp.exp(
        -0.5 * jnp.abs(beta_i - dp + d) ** 2
        + 1.0j * aOmegab(dp, beta_i)
        + 1.0j * aOmegab(d, beta_i - dp)
    )
    return jnp.abs(jnp.sum(prefactor * exponential)) ** 2


@jax.jit
def analytic_fidelity_transfer(
    alpha: Array,
    beta: Array,
    c: Array,
    d: Array,
    cp: Array,
    dp: Array,
) -> Array:
    """Batched state-transfer fidelity over all Kraus branches.

    Parameters
    ----------
    alpha, beta : Array, shape ``(K, N_l)``
    c, d : Array
        Initial state.
    cp, dp : Array
        Target state.

    Returns
    -------
    Array
        Scalar fidelity.
    """

    def body_i(i: int, acc: Array) -> Array:
        return acc + analytic_fidelity_transfer_i(alpha[i], beta[i], c, d, cp, dp)

    return jax.lax.fori_loop(0, alpha.shape[0], body_i, 0.0)


@jax.jit
def analytic_fidelity_fock_state(alphas: Array, betas: Array, m: int) -> float:
    r"""Fidelity with a pure Fock state :math:`|m\rangle`.

    Computes :math:`F_m = \sum_j |\sum_i \alpha_{ji} \langle m|\beta_{ji}\rangle|^2`.

    Parameters
    ----------
    alphas : Array, shape ``(2, N_l)``
        Complex amplitudes from :func:`~coherax.circuits.g`.
    betas : Array, shape ``(2, N_l)``
        Complex displacement positions from :func:`~coherax.circuits.g`.
    m : int
        Target Fock state number.

    Returns
    -------
    float
        Fidelity in ``[0, 1]``.
    """
    envelope = jnp.exp(-0.5 * jnp.abs(betas) ** 2)  # (2, N_l)
    monomial = betas**m  # (2, N_l), complex power
    norm = 1.0 / jnp.sqrt(jnp.exp(jsp.gammaln(m + 1.0)))
    overlaps = envelope * monomial * norm  # (2, N_l)

    # \Sigma_i \alpha_{j,i} <m|\beta_{j,i}> for each j
    inner = jnp.sum(alphas * overlaps, axis=1)  # (2,)
    return jnp.sum(jnp.abs(inner) ** 2).real


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=2)
def analytic_fidelity_wrapper(
    coherent: "CoherentKet",  # noqa: F821
    circuit_params: Array,
    N_l: int,
) -> Array:
    """Fidelity between a :class:`~coherax.states.CoherentKet` and a circuit output.

    Parameters
    ----------
    coherent : CoherentKet
        Target state.
    circuit_params : Array, shape ``(n_layers, 4)``
    N_l : int
        Coherent-term count.

    Returns
    -------
    Array
        Scalar fidelity.
    """
    from coherax.circuits import g

    alpha_coherent = jnp.expand_dims(coherent.cs, 0)
    beta_coherent = jnp.expand_dims(coherent.ds, 0)
    alpha_circuit, beta_circuit = g(circuit_params, N_l)
    return analytic_fidelity(
        all_coeffs_a=alpha_coherent,
        all_coeffs_b=alpha_circuit,
        all_peaks_a=beta_coherent,
        all_peaks_b=beta_circuit,
    )


@partial(jax.jit, static_argnums=(3, 4))
def analytic_fidelity_transfer_wrapper(
    initial: "CoherentKet",  # noqa: F821
    final: "CoherentKet",  # noqa: F821
    circuit_params: Array,
    N_l: int,
    T: int,
):
    """State-transfer fidelity from *initial* to *final* via a circuit.

    Parameters
    ----------
    initial : CoherentKet
        Starting state.
    final : CoherentKet
        Target state.
    circuit_params : Array, shape ``(T, n_layers, 4)``
    N_l : int
        Coherent-term count.
    T : int
        Number of independent circuit branches.

    Returns
    -------
    Array
        Scalar fidelity.
    """
    from coherax.circuits import super_g

    alpha, beta = super_g(circuit_params, N_l=N_l, T=T)
    return analytic_fidelity_transfer(
        alpha=alpha, beta=beta, c=initial.cs, d=initial.ds, cp=final.cs, dp=final.ds
    )


@partial(jax.jit, static_argnums=2)
def analytic_fidelity_fock_wrapper(fock_m: int, circuit_params: Array, N_l: int):
    r"""Fidelity between a circuit output and a pure Fock state :math:`|m\rangle`.

    Parameters
    ----------
    fock_m : int
        Target Fock state.
    circuit_params : Array, shape ``(n_layers, 4)``
    N_l : int
        Coherent-term count.

    Returns
    -------
    Array
        Scalar fidelity.
    """
    from coherax.circuits import g

    alpha_circuit, beta_circuit = g(circuit_params, N_l)
    return analytic_fidelity_fock_state(alpha_circuit, beta_circuit, fock_m)


# ---------------------------------------------------------------------------
# Simplified fidelity functions using inner-product formulation
# ---------------------------------------------------------------------------


def state_fidelity(psi: "Ket", phi: "Ket") -> Array:
    r"""Fidelity :math:`|\langle\psi|\phi\rangle|^2` between two kets.

    Works for any combination of :class:`~coherax.states.CoherentKet`,
    :class:`~coherax.states.FockKet`, etc.

    .. math::

        F = |\langle\psi|\phi\rangle|^2

    Parameters
    ----------
    psi : Ket
        First state.
    phi : Ket
        Second state.

    Returns
    -------
    Array
        Scalar fidelity in :math:`[0, 1]`.
    """
    return jnp.abs(psi.inner(phi)) ** 2


def circuit_state_fidelity(
    target: "CoherentKet",
    circuit_params: Array,
    N_l: int,
) -> Array:
    r"""Fidelity between a target state and a circuit output.

    Constructs the circuit unitary from *circuit_params*, applies it to
    the vacuum :math:`|0\rangle|0\rangle`, traces out the qubit by
    projecting onto :math:`|0\rangle`, and computes the overlap with
    *target*.

    .. math::

        F = \bigl|\langle\mathrm{target}|\,
            \langle 0|\, U\,|0\rangle|0\rangle\bigr|^2

    Parameters
    ----------
    target : CoherentKet
        Target bosonic state.
    circuit_params : Array, shape ``(n_layers, 4)``
        Circuit layer parameters.
    N_l : int
        Coherent-term count.

    Returns
    -------
    Array
        Scalar fidelity.
    """
    from coherax.circuits import CircuitUnitary
    from coherax.states import CoherentKet, JointKet, QubitKet

    U = CircuitUnitary.from_params(circuit_params, N_l)
    vacuum = CoherentKet(cs=jnp.array([1.0]), ds=jnp.array([0.0 + 0j]))
    qubit0 = QubitKet(cs=jnp.array([1.0, 0.0]))
    output = U.apply(vacuum, qubit0)
    # Project onto qubit |0> to get the bosonic output
    bosonic_output = output.inner(qubit0)
    return jnp.abs(target.inner(bosonic_output)) ** 2


def circuit_fock_fidelity(
    m: int,
    circuit_params: Array,
    N_l: int,
) -> Array:
    r"""Fidelity between a circuit output and Fock state :math:`|m\rangle`.

    .. math::

        F = \bigl|\langle m|\,
            \langle 0|\, U\,|0\rangle|0\rangle\bigr|^2

    Parameters
    ----------
    m : int
        Target Fock state number.
    circuit_params : Array, shape ``(n_layers, 4)``
    N_l : int
        Coherent-term count.

    Returns
    -------
    Array
        Scalar fidelity.
    """
    from coherax.circuits import CircuitUnitary
    from coherax.states import CoherentKet, FockKet, QubitKet

    U = CircuitUnitary.from_params(circuit_params, N_l)
    vacuum = CoherentKet(cs=jnp.array([1.0]), ds=jnp.array([0.0 + 0j]))
    qubit0 = QubitKet(cs=jnp.array([1.0, 0.0]))
    output = U.apply(vacuum, qubit0)
    bosonic_output = output.inner(qubit0)
    fock_target = FockKet(cs=jnp.array([1.0]), ns=jnp.array([m]))
    return jnp.abs(fock_target.inner(bosonic_output)) ** 2


# TODO general focks states
