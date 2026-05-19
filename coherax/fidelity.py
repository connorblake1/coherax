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

from coherax.linalg_utils import coherent_overlap, dag, invsqrtm_supp, aOmegab
from coherax.circuits import CircuitUnitary, g, super_g
from coherax.states import (
    CoherentKet,
    FockKet,
    Ket,
    QubitKet,
    _beamsplit_full_arrays,
    unitary_encoding_map,
)


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
    coherent: CoherentKet,
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
    initial: CoherentKet,
    final: CoherentKet,
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
    alpha_circuit, beta_circuit = g(circuit_params, N_l)
    return analytic_fidelity_fock_state(alpha_circuit, beta_circuit, fock_m)


# ---------------------------------------------------------------------------
# Simplified fidelity functions using inner-product formulation
# ---------------------------------------------------------------------------


def state_fidelity(psi: Ket, phi: Ket) -> Array:
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
    target: CoherentKet,
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
    U = CircuitUnitary.from_params(circuit_params, N_l)
    vacuum = CoherentKet(cs=jnp.array([1.0]), ds=jnp.array([0.0 + 0j]))
    qubit0 = QubitKet(cs=jnp.array([1.0, 0.0]))
    output = U.apply(vacuum, qubit0)
    bosonic_output = output.inner(qubit0)
    fock_target = FockKet(cs=jnp.array([1.0]), ns=jnp.array([m]))
    return jnp.abs(fock_target.inner(bosonic_output)) ** 2


# ---------------------------------------------------------------------------
# Floating-basis: entanglement fidelity & coherent information under pure loss
#
# All-coherent-basis formulation: encode by C = G^{-1/2} X (X^dag X)^{-1/2},
# transmit through a beamsplitter to a vacuum environment, then either
# (a) decode with a CPTP Kraus channel parametrized by Z (for F_e), or
# (b) compute the entropies of rho_B and rho_RB directly in the coherent
# basis using the Loewner-equivalent rho_tilde = G^{1/2} rho G^{1/2} (for I_c).
#
# The encoder X: (A, D) and decoder Z: (N_D, D, A) are unconstrained complex
# arrays; the algebraic isometry / Kraus normalization is baked into the
# objective so no constraint enforcement is required.
# ---------------------------------------------------------------------------


_VAC_Y = jnp.array([1.0 + 0j])
_VAC_SIGMA = jnp.array([0.0 + 0j])


@jax.jit
def _cptp_decoding_map(Z: Array, d_out: Array, rho_loss: Array) -> Array:
    r"""Apply a CPTP decoder to a coherent-basis post-loss density tensor.

    Kraus operators are :math:`D_i = \sum_{k,a} B_{ika} |k\rangle\langle
    \phi^{(D)}_a|`, with :math:`B = Z\,S^{-1/2}|_{\mathrm{supp}}` where
    :math:`S_{ab} = \sum_i Z_i^* Z_i` enforces
    :math:`\sum_i D_i^\dagger D_i = I` algebraically. The conversion from
    the non-orthogonal coherent basis at the channel output to the
    physical orthonormal basis uses :math:`T^{-1}_\dagger =
    G^{1/2}|_{\mathrm{supp}}\,U_G^\dagger`.

    Parameters
    ----------
    Z : Array, shape ``(N_D, D, A)``
    d_out : Array, shape ``(A,)``
    rho_loss : Array, shape ``(D, D, A, A)``

    Returns
    -------
    Array, shape ``(D, D, D, D)``
    """
    A = d_out.shape[0]
    S = jnp.einsum("ika,ikb->ab", jnp.conj(Z), Z)
    B = jnp.einsum("ikb,ba->ika", Z, invsqrtm_supp(S))
    G = coherent_overlap(d_out.reshape((A, 1)), d_out.reshape((1, A)))
    w, U = jnp.linalg.eigh(G)
    w_pos = jnp.maximum(jnp.real(w), 1e-10)
    Tinv = jnp.diag(jnp.sqrt(w_pos)) @ dag(U)
    C = jnp.einsum("pki,ia->pka", B, Tinv)
    Gamma = jnp.einsum("pka,plb->kalb", C, jnp.conj(C))
    return jnp.einsum("kalb,uvab->uvkl", Gamma, rho_loss)


@jax.jit
def entanglement_fidelity_pureloss(
    X: Array,
    d: Array,
    Z: Array,
    gamma: float | Array,
) -> Array:
    r"""Entanglement fidelity of an encode-loss-decode chain (floating basis).

    .. math::

        F_e = \frac{1}{D^2}\sum_{\mu,\nu=0}^{D-1}
              \langle\mu|\,(\mathcal{D}\!\circ\!\mathcal{E}\!\circ\!\mathcal{C})
              (|\mu\rangle\!\langle\nu|)\,|\nu\rangle

    where :math:`\mathcal{C}` is the floating-basis encoder
    :func:`~coherax.states.unitary_encoding_map`, :math:`\mathcal{E}` is
    pure photon loss with rate :math:`\gamma`, and :math:`\mathcal{D}` is
    the CPTP decoder parametrized by :math:`Z`.

    Parameters
    ----------
    X : Array, shape ``(A, D)``
        Encoder parameter matrix.
    d : Array, shape ``(A,)``
        Coherent-state displacements.
    Z : Array, shape ``(N_D, D, A)``
        Unconstrained decoder Kraus parameters.
    gamma : float or Array
        Pure-loss rate in :math:`[0, 1)`.

    Returns
    -------
    Array
        Real scalar :math:`F_e \in [0, 1]`.
    """
    D = X.shape[1]
    eta = 1.0 - gamma
    encoded = jax.vmap(lambda mu: unitary_encoding_map(X, d, mu))(
        jnp.eye(D, dtype=jnp.complex128)
    )  # (D, A)
    rho_loss, d_out = _beamsplit_full_arrays(
        encoded, d, _VAC_Y, _VAC_SIGMA, jnp.asarray(eta)
    )
    rho_dec = _cptp_decoding_map(Z, d_out, rho_loss)  # (D, D, D, D)
    Fe = jnp.einsum("mnmn->", rho_dec)
    return jnp.real(Fe) / (D**2)


def _entropy_nats(evals: Array) -> Array:
    r"""Von Neumann entropy in nats, gradient-safe at zero."""
    evals = jnp.real(evals)
    safe = jnp.maximum(evals, 1e-30)
    return -jnp.sum(jnp.where(evals > 1e-15, evals * jnp.log(safe), 0.0))


@jax.jit
def coherent_information_pureloss(
    X: Array,
    d: Array,
    gamma: float | Array,
) -> Array:
    r"""Coherent information of a floating-basis encoder under pure loss (qubits).

    Computes :math:`I_c = S(\rho_B) - S(\rho_{RB})` for the maximally
    entangled reference state passed through the encoder + loss channel,
    entirely in the coherent basis. The Loewner-equivalent density matrix
    :math:`\tilde\rho = G^{1/2}\,\rho_{\mathrm{coh}}\,G^{1/2}` has the
    same nonzero spectrum as the physical :math:`\rho`, avoiding any Fock
    truncation. Result is returned in qubits (entropies divided by
    :math:`\ln 2`).

    Parameters
    ----------
    X : Array, shape ``(A, D)``
        Encoder parameter matrix.
    d : Array, shape ``(A,)``
        Coherent-state displacements.
    gamma : float or Array
        Pure-loss rate in :math:`[0, 1)`.

    Returns
    -------
    Array
        :math:`I_c` in qubits.
    """
    D = X.shape[1]
    eta = 1.0 - gamma
    encoded = jax.vmap(lambda mu: unitary_encoding_map(X, d, mu))(
        jnp.eye(D, dtype=jnp.complex128)
    )  # (D, A)
    rho_out, d_out = _beamsplit_full_arrays(
        encoded, d, _VAC_Y, _VAC_SIGMA, jnp.asarray(eta)
    )
    A = d_out.shape[0]
    G = coherent_overlap(d_out.reshape((A, 1)), d_out.reshape((1, A)))
    w, U = jnp.linalg.eigh(G)
    w_pos = jnp.maximum(jnp.real(w), 1e-12)
    G_half = U @ jnp.diag(jnp.sqrt(w_pos)) @ dag(U)
    # rho_B = (1/D) sum_mu rho_out[mu, mu]; Loewner-conjugate by G_half
    rho_B = jnp.einsum("mmab->ab", rho_out) / D
    rho_B = G_half @ rho_B @ G_half
    rho_B = (rho_B + dag(rho_B)) / 2
    S_B = _entropy_nats(jnp.linalg.eigvalsh(rho_B))
    # rho_RB: block-reshape (D, D, A, A) -> (D*A, D*A) and Loewner-conjugate
    rho_RB = rho_out.transpose(0, 2, 1, 3).reshape(D * A, D * A) / D
    G_RB = jnp.kron(jnp.eye(D), G_half)
    rho_RB = G_RB @ rho_RB @ G_RB
    rho_RB = (rho_RB + dag(rho_RB)) / 2
    S_RB = _entropy_nats(jnp.linalg.eigvalsh(rho_RB))
    return (S_B - S_RB) / jnp.log(2.0)


@jax.jit
def nbar_logical(X: Array, d: Array) -> Array:
    r"""Mean photon number of the maximally mixed encoded logical state.

    For the encoded basis :math:`\{|\psi_\mu\rangle\}` and the
    maximally mixed logical state :math:`\rho = I_D / D`, returns

    .. math::

        \bar n = \frac{1}{D}\sum_\mu
                 \langle\psi_\mu | \hat n | \psi_\mu\rangle,
        \qquad
        \hat n = \sum_{ab} d_a^*\,d_b\,G_{ab}\,|d_a\rangle\!\langle d_b|.

    Parameters
    ----------
    X : Array, shape ``(A, D)``
    d : Array, shape ``(A,)``

    Returns
    -------
    Array
        Real scalar mean photon number.
    """
    A = d.shape[0]
    D = X.shape[1]
    G = coherent_overlap(d.reshape((A, 1)), d.reshape((1, A)))
    nhat = jnp.conj(d).reshape((A, 1)) * d.reshape((1, A)) * G
    encoded = jax.vmap(lambda mu: unitary_encoding_map(X, d, mu))(
        jnp.eye(D, dtype=jnp.complex128)
    )  # (D, A)
    # <psi_mu|nhat|psi_mu> = sum_ab conj(c_a) nhat_ab c_b
    expected = jnp.einsum("ma,ab,mb->", jnp.conj(encoded), nhat, encoded)
    return jnp.real(expected) / D
