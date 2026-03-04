"""Analytic fidelity computations in the coherent basis.

All functions operate on the ``(alpha, beta)`` representation produced by
:func:`coherax.circuits.g` and compute fidelities without Fock-space
simulation.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array

from coherax.operators import aOmegab


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


# ---------------------------------------------------------------------------
# Pure-loss recovery fidelity
# ---------------------------------------------------------------------------


@jax.jit
def analytic_pureloss_recovery_fidelity_thetaphi_iab(
    da: Array,
    db: Array,
    alpha_i: Array,
    beta_i: Array,
    cap: Array,
    dap: Array,
    gamma: float,
) -> Array:
    r"""Pure-loss recovery fidelity kernel for a specific ``(a, b)`` index pair.

    This is the innermost computation of the analytic fidelity under
    photon loss at rate ``gamma``.

    Parameters
    ----------
    da, db : Array
        Displacement amplitudes of code-word components.
    alpha_i, beta_i : Array, shape ``(N_l,)``
        Recovery channel coefficients and displacements.
    cap, dap : Array, shape ``(A,)``
        Input state coefficients and displacements.
    gamma : float
        Loss probability.

    Returns
    -------
    Array
        Complex amplitude contribution.
    """
    A = cap.shape[0]
    N = alpha_i.shape[0]
    cap = cap.reshape(A, 1)
    dap = dap.reshape(A, 1)
    alpha_i = alpha_i.reshape(1, N)
    beta_i = beta_i.reshape(1, N)

    prefactor = jnp.conj(cap) * alpha_i
    env_term1 = (-1.0 + jnp.sqrt(1 - gamma)) / 2.0 * jnp.abs(beta_i - dap) ** 2
    env_term2 = (
        -jnp.sqrt(1 - gamma) / 2.0
        * (jnp.abs(beta_i - dap + db) ** 2 - jnp.abs(db) ** 2)
    )
    env_term3 = -0.25 * (
        gamma * jnp.abs(da - db) ** 2
        + (1.0 - gamma) * (jnp.abs(da) ** 2 + jnp.abs(db) ** 2)
    )
    envelope = env_term1 + env_term2 + env_term3
    phase = 1.0j * (
        aOmegab(dap, beta_i)
        + jnp.sqrt(1 - gamma) * aOmegab(db, beta_i - dap)
        + gamma / 2 * aOmegab(da, db)
    )
    return jnp.sum(prefactor * jnp.exp(envelope + phase))


@jax.jit
def analytic_pureloss_recovery_fidelity_thetaphi(
    alpha: Array,
    beta: Array,
    c: Array,
    d: Array,
    gamma: float,
) -> Array:
    r"""Fidelity of a CD+R recovery circuit under pure loss.

    Parameters
    ----------
    alpha, beta : Array, shape ``(K, N_l)``
        Recovery channel representation.
    c, d : Array
        Logical state coefficients and displacements.
    gamma : float
        Loss probability.

    Returns
    -------
    Array
        Scalar fidelity.
    """
    partial_i_fidelity_caller = jax.jit(
        partial(
            jax.vmap(
                jax.vmap(
                    lambda da, db, alpha_i, beta_i: (
                        analytic_pureloss_recovery_fidelity_thetaphi_iab(
                            da, db, alpha_i, beta_i, c, d, gamma
                        )
                        * jnp.conj(
                            analytic_pureloss_recovery_fidelity_thetaphi_iab(
                                db, da, alpha_i, beta_i, c, d, gamma
                            )
                        )
                    ),
                    in_axes=(None, 0, None, None),
                ),
                in_axes=(0, None, None, None),
            ),
            d,
            d,
        )
    )

    def body_i(i: int, acc: Array) -> Array:
        return acc + jnp.abs(
            jnp.einsum(
                "ij,i,j->", partial_i_fidelity_caller(alpha[i], beta[i]), jnp.conj(c), c
            )
        )

    return jax.lax.fori_loop(0, alpha.shape[0], body_i, 0.0)


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=2)
def analytic_fidelity_wrapper(
    coherent: "CoherentKet",  # noqa: F821
    circuit_parameters: Array,
    N_l: int,
) -> Array:
    """Fidelity between a :class:`~coherax.states.CoherentKet` and a circuit output.

    Parameters
    ----------
    coherent : CoherentKet
        Target state.
    circuit_parameters : Array, shape ``(n_layers, 4)``
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
    alpha_circuit, beta_circuit = g(circuit_parameters, N_l)
    return analytic_fidelity(
        all_coeffs_a=alpha_coherent,
        all_coeffs_b=alpha_circuit,
        all_peaks_a=beta_coherent,
        all_peaks_b=beta_circuit,
    )


