"""CD+R circuit construction and coherent-basis channel representation.

Provides unitaries (CD, ECD, qubit rotations), the :class:`TraceoutLayer`
for tracking coherent-basis amplitudes through a circuit, and the core
function :func:`g` that extracts the ``(alpha, beta)`` representation from
circuit parameters.
"""

from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from jaxtyping import Array

from coherax.operators import (
    GKP_N,
    IN,
    I2,
    a_op,
    a_dag_op,
    aOmegab,
    dqdag,
    dqdisplace,
    dqtensor,
    e_n1iaOmegab,
    sigma_x,
    sigma_y,
    sigma_z,
)

import dynamiqs as dq

# ---------------------------------------------------------------------------
# Unitaries
# ---------------------------------------------------------------------------


@jax.jit
def W(u: complex) -> Array:
    r"""Displacement operator :math:`D(u\sqrt{\pi})`.

    Parameters
    ----------
    u : complex
        Scaled displacement amplitude.
    """
    return dqdisplace(GKP_N, u * jnp.sqrt(jnp.pi))


@jax.jit
def CD(u: complex) -> Array:
    r"""Controlled displacement :math:`e^{(u \hat{a}^\dagger - u^* \hat{a}) \otimes \sigma_z}`.

    Parameters
    ----------
    u : complex
        Displacement amplitude.
    """
    return jla.expm(dqtensor(u * a_dag_op - jnp.conj(u) * a_op, sigma_z))


@jax.jit
def ECD(beta: complex) -> Array:
    r"""Echoed controlled displacement: :math:`(\mathbb{I} \otimes \sigma_x) \cdot CD(\beta/2)`.

    Parameters
    ----------
    beta : complex
        Displacement amplitude.
    """
    return dqtensor(IN, sigma_x) @ CD(beta / 2)


def R_x(theta: float) -> Array:
    r"""Qubit rotation about the *x*-axis by angle *theta*."""
    return jla.expm(-1j * theta * sigma_x / 2)


def R_y(theta: float) -> Array:
    r"""Qubit rotation about the *y*-axis by angle *theta*."""
    return jla.expm(-1j * theta * sigma_y / 2)


def R_z(theta: float) -> Array:
    r"""Qubit rotation about the *z*-axis by angle *theta*."""
    return jla.expm(-1j * theta * sigma_z / 2)


@jax.jit
def qubit_rotation(phi: float, theta: float, gamma: float) -> Array:
    r"""General qubit rotation :math:`R_z(\phi) R_y(\theta) R_z(\gamma)`.

    Parameters
    ----------
    phi, theta, gamma : float
        Euler angles.

    Returns
    -------
    Array, shape ``(2, 2)``
    """
    return jnp.exp(-0.5j * (gamma + phi)) * jnp.array(
        [
            [jnp.cos(theta / 2), -jnp.sin(theta / 2) * jnp.exp(1.0j * gamma)],
            [
                jnp.sin(theta / 2) * jnp.exp(1.0j * phi),
                jnp.cos(theta / 2) * jnp.exp(1.0j * (phi + gamma)),
            ],
        ]
    )


@jax.jit
def ecd_rotation_2x2(phi: float, theta: float, gamma: float) -> Array:
    r"""Qubit rotation matrix in the ECD convention (rows swapped).

    Parameters
    ----------
    phi, theta, gamma : float
        Euler angles.

    Returns
    -------
    Array, shape ``(2, 2)``
    """
    return jnp.exp(-0.5j * (gamma + phi)) * jnp.array(
        [
            [
                jnp.sin(theta / 2) * jnp.exp(1.0j * phi),
                jnp.cos(theta / 2) * jnp.exp(1.0j * (phi + gamma)),
            ],
            [jnp.cos(theta / 2), -jnp.sin(theta / 2) * jnp.exp(1.0j * gamma)],
        ]
    )


# ---------------------------------------------------------------------------
# Circuit layers
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=1)
def circuit_layer(layer: Array, N: int = GKP_N) -> Array:
    """Single ECD + qubit-rotation layer.

    Parameters
    ----------
    layer : Array, shape ``(4,)``
        ``[displacement, phi, theta, gamma]`` (displacement is complex).
    N : int
        Hilbert space dimension.

    Returns
    -------
    Array, shape ``(2N, 2N)``
        Unitary on the joint cavity ⊗ qubit space.
    """
    d = layer[0]
    phi = jnp.real(layer[1])
    theta = jnp.real(layer[2])
    gamma = jnp.real(layer[3])
    return ECD(beta=d) @ dqtensor(jnp.eye(N), qubit_rotation(phi=phi, theta=theta, gamma=gamma))


@jax.jit
def compose_ECD_layers(params: Array) -> Array:
    """Compose a sequence of ECD layers into a single unitary.

    Parameters
    ----------
    params : Array, shape ``(n_layers, 4)``
        Each row is ``[displacement, phi, theta, gamma]``.

    Returns
    -------
    Array, shape ``(2*GKP_N, 2*GKP_N)``
        Composed unitary.
    """
    circ = circuit_layer(params[0, :])

    def body_mult(i: int, c: Array) -> Array:
        return circuit_layer(params[i, :]) @ c

    return jax.lax.fori_loop(1, params.shape[0], body_mult, circ)


@partial(jax.jit, static_argnums=1)
def traceout_unitary(U: Array, N: int = GKP_N) -> Array:
    """Trace out the qubit from a cavity⊗qubit unitary to get Kraus operators.

    Parameters
    ----------
    U : Array, shape ``(2N, 2N)``
        Joint unitary.
    N : int
        Cavity dimension.

    Returns
    -------
    Array, shape ``(2, N, N)``
        Two Kraus operators corresponding to qubit measurement outcomes.
    """
    K = jnp.zeros((2, N, N), jnp.complex64)
    K = K.at[0, :, :].set(
        dqtensor(jnp.eye(N), dqdag(dq.fock(2, 0))) @ U @ dqtensor(jnp.eye(N), dq.fock(2, 0))
    )
    K = K.at[1, :, :].set(
        dqtensor(jnp.eye(N), dqdag(dq.fock(2, 1))) @ U @ dqtensor(jnp.eye(N), dq.fock(2, 0))
    )
    return K


@jax.jit
def circuit_params_to_2channel(params: Array) -> Array:
    """Convert circuit parameters to a 2-element Kraus channel.

    Parameters
    ----------
    params : Array, shape ``(n_layers, 4)``

    Returns
    -------
    Array, shape ``(2, GKP_N, GKP_N)``
    """
    return traceout_unitary(compose_ECD_layers(params))


# ---------------------------------------------------------------------------
# TraceoutLayer — coherent-basis channel tracking
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=1)
def _addmask(n: int, N_l: int) -> Array:
    index = 2**n
    mask = jnp.arange(N_l) < index
    return mask.astype(jnp.complex64)


@partial(jax.jit, static_argnums=1)
def _caddmask(n: int, N_l: int) -> Array:
    return jnp.roll(_addmask(n, N_l), 2**n, axis=0)


class TraceoutLayer(eqx.Module):
    r"""Coherent-basis representation of a traced-out CD+R circuit.

    Tracks amplitudes ``(alphas, betas)`` through sequential ECD layers,
    enabling analytic fidelity computation without Fock-space simulation.

    Parameters
    ----------
    n : int
        Number of composed layers.
    N_l : int
        Coherent-term count (:math:`2^{\text{depth}}`).
    alphas : Array or None
        Coefficient tensor, shape ``(2, 2, N_l)``.
    betas : Array or None
        Displacement tensor, shape ``(2, 2, N_l)``.
    """

    alphas: Array
    betas: Array
    n: int
    N_l: int

    def __init__(
        self,
        n: int,
        N_l: int,
        alphas: Array | None = None,
        betas: Array | None = None,
    ) -> None:
        self.n = n
        self.N_l = N_l
        self.alphas = alphas if alphas is not None else jnp.zeros((2, 2, N_l), jnp.complex64)
        self.betas = betas if betas is not None else jnp.zeros((2, 2, N_l), jnp.complex64)

    @staticmethod
    @partial(jax.jit, static_argnums=1)
    def from_single_param(circuit_layer_params: Array, N_l: int) -> TraceoutLayer:
        """Construct from a single circuit layer's parameters.

        Parameters
        ----------
        circuit_layer_params : Array, shape ``(4,)``
            ``[displacement, phi, theta, gamma]``.
        N_l : int
            Coherent-term count.
        """
        d = circuit_layer_params[0].astype(jnp.complex64)
        phi = jnp.real(circuit_layer_params[1])
        theta = jnp.real(circuit_layer_params[2])
        gamma = jnp.real(circuit_layer_params[3])
        alphas = jnp.zeros((2, 2, N_l), dtype=jnp.complex64)
        betas = jnp.zeros((2, 2, N_l), dtype=jnp.complex64)
        alphas = alphas.at[:, :, 0].set(ecd_rotation_2x2(phi=phi, theta=theta, gamma=gamma))
        betas = betas.at[:, :, 0].set(jnp.array([[-d / 2, -d / 2], [d / 2, d / 2]]))
        return TraceoutLayer(n=1, N_l=N_l, alphas=alphas, betas=betas)

    @staticmethod
    @partial(jax.jit, static_argnums=2)
    def unitarycompose(l_a: TraceoutLayer, l_b: TraceoutLayer, N_l: int) -> TraceoutLayer:
        """Compose a single-layer ``l_a`` onto a multi-layer ``l_b``.

        Parameters
        ----------
        l_a : TraceoutLayer
            Must be a single layer (``n=1``).
        l_b : TraceoutLayer
            Accumulated layers.
        N_l : int
            Coherent-term count.
        """
        n_b = l_b.n - 1
        n_add = _addmask(n_b, N_l)
        n_cadd = _caddmask(n_b, N_l)
        full_add = _addmask(n_b + 1, N_l)
        cshift = lambda x: jnp.roll(x, 2**n_b, axis=0)

        d = l_b.betas
        c = l_b.alphas
        alphas = l_a.alphas
        betas = l_a.betas

        d2_minus = betas[0, 0, 0] * full_add
        d2_plus = betas[1, 1, 0] * full_add

        dcol0 = n_add * d[0, 0, :] + n_cadd * cshift(d[1, 0, :])
        dcol1 = n_add * d[0, 1, :] + n_cadd * cshift(d[1, 1, :])

        out_betas = jnp.array(
            [[d2_minus + dcol0, d2_minus + dcol1], [d2_plus + dcol0, d2_plus + dcol1]]
        )

        c00 = (
            alphas[0, 0, 0] * n_add * c[0, 0, :] * e_n1iaOmegab(betas[0, 0, 0] * full_add, n_add * d[0, 0, :])
        ) + cshift(
            alphas[0, 1, 0] * n_add * c[1, 0, :] * e_n1iaOmegab(betas[0, 1, 0] * n_add, d[1, 0, :])
        )
        c01 = (
            alphas[0, 0, 0] * n_add * c[0, 1, :] * e_n1iaOmegab(betas[0, 0, 0] * full_add, n_add * d[0, 1, :])
        ) + cshift(
            alphas[0, 1, 0] * n_add * c[1, 1, :] * e_n1iaOmegab(betas[0, 1, 0] * n_add, d[1, 1, :])
        )
        c10 = (
            alphas[1, 0, 0] * n_add * c[0, 0, :] * e_n1iaOmegab(betas[1, 0, 0] * full_add, n_add * d[0, 0, :])
        ) + cshift(
            alphas[1, 1, 0] * n_add * c[1, 0, :] * e_n1iaOmegab(betas[1, 1, 0] * n_add, d[1, 0, :])
        )
        c11 = (
            alphas[1, 0, 0] * n_add * c[0, 1, :] * e_n1iaOmegab(betas[1, 0, 0] * full_add, n_add * d[0, 1, :])
        ) + cshift(
            alphas[1, 1, 0] * n_add * c[1, 1, :] * e_n1iaOmegab(betas[1, 1, 0] * n_add, d[1, 1, :])
        )

        out_alphas = jnp.array([[c00, c01], [c10, c11]], dtype=jnp.complex64)
        return TraceoutLayer(n=l_b.n + 1, N_l=alphas.shape[2], alphas=out_alphas, betas=out_betas)

    @staticmethod
    @jax.jit
    def to_traceout(layer: TraceoutLayer) -> tuple[Array, Array]:
        """Extract the ``(alphas, betas)`` pair from a composed layer.

        Returns
        -------
        alphas : Array, shape ``(2, N_l)``
        betas : Array, shape ``(2, N_l)``
        """
        return layer.alphas[:, 0, :], layer.betas[:, 0, :]

    @staticmethod
    @partial(jax.jit, static_argnums=1)
    def from_params(circuit_params: Array, N_l: int) -> TraceoutLayer:
        """Compose all circuit layers into a single :class:`TraceoutLayer`.

        Parameters
        ----------
        circuit_params : Array, shape ``(n_layers, 4)``
        N_l : int
            Coherent-term count (:math:`2^{n_{\\text{layers}}}`).
        """
        circ = TraceoutLayer.from_single_param(circuit_layer_params=circuit_params[0, :], N_l=N_l)

        @jax.jit
        def body_compose(i: int, c: TraceoutLayer) -> TraceoutLayer:
            return TraceoutLayer.unitarycompose(
                l_a=TraceoutLayer.from_single_param(circuit_layer_params=circuit_params[i, :], N_l=N_l),
                l_b=c,
                N_l=N_l,
            )

        return jax.lax.fori_loop(1, circuit_params.shape[0], body_fun=body_compose, init_val=circ)


# ---------------------------------------------------------------------------
# Core extraction functions
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=1)
def g(circuit_params: Array, N_l: int) -> tuple[Array, Array]:
    r"""Extract ``(alphas, betas)`` coherent representation from circuit parameters.

    This is the key function that converts a sequence of CD+R layer
    parameters into the analytic coherent-basis representation used for
    all downstream fidelity computations.

    Parameters
    ----------
    circuit_params : Array, shape ``(n_layers, 4)``
        Each row: ``[displacement, phi, theta, gamma]``.
    N_l : int
        Coherent-term count (must be :math:`2^{n_{\\text{layers}}}`).

    Returns
    -------
    alphas : Array, shape ``(2, N_l)``
        Coherent-basis coefficients per Kraus branch.
    betas : Array, shape ``(2, N_l)``
        Displacement amplitudes per Kraus branch.
    """
    return TraceoutLayer.to_traceout(TraceoutLayer.from_params(circuit_params, N_l))


def channel_from_b(alphas: Array, betas: Array) -> Array:
    """Convert coherent ``(alpha, beta)`` representation to Fock-basis Kraus operators.

    Parameters
    ----------
    alphas : Array, shape ``(K, M)``
        Coefficients.
    betas : Array, shape ``(K, M)``
        Displacement amplitudes.

    Returns
    -------
    Array, shape ``(K, GKP_N, GKP_N)``
        Kraus operators in the Fock basis.
    """
    ops = jnp.zeros((alphas.shape[0], GKP_N, GKP_N), dtype=jnp.complex64)

    @jax.jit
    def sum_displacements(j: int) -> Array:
        @jax.jit
        def body_fun(i: int, partial_sum: Array) -> Array:
            return partial_sum + alphas[j, i] * dqdisplace(GKP_N, betas[j, i]).astype(jnp.complex64)

        return jax.lax.fori_loop(0, alphas.shape[1], body_fun, jnp.zeros((GKP_N, GKP_N), dtype=jnp.complex64))

    @jax.jit
    def outer_body(j: int, ops_acc: Array) -> Array:
        return ops_acc.at[j, :, :].set(sum_displacements(j))

    return jax.lax.fori_loop(0, alphas.shape[0], outer_body, ops)


@partial(jax.jit, static_argnums=(1, 2))
def super_g(super_circuit_params: Array, N_l: int, T: int) -> tuple[Array, Array]:
    r"""Extend :func:`g` to a superposition of *T* independent circuits.

    Parameters
    ----------
    super_circuit_params : Array, shape ``(T, n_layers, 4)``
        Parameters for *T* circuits.
    N_l : int
        Coherent-term count per circuit.
    T : int
        Number of circuits in the superposition.

    Returns
    -------
    alphas : Array, shape ``(2^T, N_l^T)``
    betas : Array, shape ``(2^T, N_l^T)``
    """
    alphas_pre = jnp.zeros((T, 2, N_l), jnp.complex64)
    betas_pre = jnp.zeros((T, 2, N_l), jnp.complex64)
    for i in range(T):
        a, b = g(circuit_params=super_circuit_params[i], N_l=N_l)
        alphas_pre = alphas_pre.at[i, :, :].set(a)
        betas_pre = betas_pre.at[i, :, :].set(b)
    alpha_total = jnp.zeros((2**T, N_l**T), jnp.complex64)
    beta_total = jnp.zeros((2**T, N_l**T), jnp.complex64)
    N_filled = N_l
    alpha_total = alpha_total.at[:2, :N_filled].set(alphas_pre[0])
    beta_total = beta_total.at[:2, :N_filled].set(betas_pre[0])
    for i in range(1, T):
        alpha_total = alpha_total.at[: 2 ** (i + 1), : N_filled * N_l].set(
            (
                alphas_pre[i, :, None, :, None]
                * alpha_total[None, : (2**i), None, :N_filled]
                * e_n1iaOmegab(
                    betas_pre[i, :, None, :, None],
                    beta_total[None, : (2**i), None, :N_filled],
                )
            ).reshape((2 ** (i + 1), N_filled * N_l))
        )
        beta_total = beta_total.at[: 2 ** (i + 1), : N_filled * N_l].set(
            (
                betas_pre[i, :, None, :, None]
                + beta_total[None, : (2**i), None, :N_filled]
            ).reshape((2 ** (i + 1), N_filled * N_l))
        )
        N_filled = N_filled * N_l
    return alpha_total, beta_total


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

_ANCILLA_TIME: float = 24e-9
"""Ancilla reset time in seconds."""


def gate_timer(beta: Array) -> Array:
    """Estimate gate duration for a displacement of magnitude ``|beta|``.

    Uses the dispersive coupling rate :math:`\\chi = 2\\pi \\times 50\\,\\text{kHz}`
    and damping factor :math:`\\gamma_0 = 20`.

    Parameters
    ----------
    beta : Array
        Complex displacement amplitude.

    Returns
    -------
    Array
        Gate time in seconds (minimum 48 ns).
    """
    chi = 2 * jnp.pi * 5e4
    gamma_0 = 20
    return jnp.clip(jnp.abs(beta) / chi / gamma_0, min=48e-9)


def circuit_params_to_time(circuit_params: Array) -> float:
    """Estimate total circuit execution time from parameters.

    Parameters
    ----------
    circuit_params : Array, shape ``(n_layers, 4)`` or ``(T, n_layers, 4)``

    Returns
    -------
    float
        Total time in seconds.
    """
    T = 0.0
    if len(circuit_params.shape) == 2:
        circuit_params = jnp.expand_dims(circuit_params, 0)
    for i in range(circuit_params.shape[0]):
        for j in range(circuit_params.shape[1]):
            T += _ANCILLA_TIME + gate_timer(jnp.array(circuit_params[i, j, 0]))
    return T
