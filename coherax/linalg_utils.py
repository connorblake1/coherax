"""Pure-JAX math primitives used across the library.

Hilbert-space truncation constant, the analytic coherent-state inner-product
kernels, the pure-JAX adjoint, and a few linear-algebra helpers. No dynamiqs
dependency.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

# ---------------------------------------------------------------------------
# Hilbert space truncation
# ---------------------------------------------------------------------------

GKP_N: int = 100
"""Fock-space truncation dimension for the bosonic mode."""


# ---------------------------------------------------------------------------
# Adjoint
# ---------------------------------------------------------------------------


@jax.jit
def dag(arr: Array) -> Array:
    """Pure-JAX conjugate transpose."""
    return jnp.conj(arr.T)


# ---------------------------------------------------------------------------
# Symplectic geometry / coherent-state kernels
# ---------------------------------------------------------------------------


@jax.jit
def aOmegab(a: Array, b: Array) -> Array:
    r"""Symplectic inner product :math:`\operatorname{Re}(a)\operatorname{Im}(b)
    - \operatorname{Im}(a)\operatorname{Re}(b)`.

    Parameters
    ----------
    a, b : Array
        Complex-valued arrays (broadcastable).

    Returns
    -------
    Array
        Real-valued symplectic product.
    """
    return jnp.real(a) * jnp.imag(b) - jnp.imag(a) * jnp.real(b)


@jax.jit
def e_n1iaOmegab(a: Array, b: Array) -> Array:
    r"""Phase factor :math:`e^{-i\, a \Omega b}`."""
    return jnp.exp(-1j * aOmegab(a, b))


@jax.jit
def coherent_overlap(alpha: Array, beta: Array) -> Array:
    r"""Inner product :math:`\langle\alpha|\beta\rangle` of two coherent states.

    Parameters
    ----------
    alpha, beta : Array
        Complex amplitudes (broadcastable).

    Returns
    -------
    Array
        Complex overlap.
    """
    return jnp.exp(-0.5 * jnp.abs(alpha - beta) ** 2 + 1.0j * aOmegab(alpha, beta))


# ---------------------------------------------------------------------------
# Random initialization
# ---------------------------------------------------------------------------


def complex_normal(key: Array, shape: Sequence[int]) -> Array:
    r"""Draw i.i.d. complex normals with unit variance.

    Real and imaginary parts are independent normals with variance
    :math:`1/2`, so :math:`\mathbb{E}[|z|^2] = 1`.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key.
    shape : Sequence[int]
        Output array shape.

    Returns
    -------
    Array, dtype ``complex64`` (matches the JAX default; pass a
        ``complex128`` cast at the call site if double precision is
        required).
    """
    kr, ki = jr.split(key, 2)
    return (jr.normal(kr, shape) + 1.0j * jr.normal(ki, shape)) / jnp.sqrt(2.0)


# ---------------------------------------------------------------------------
# Linear-algebra helpers
# ---------------------------------------------------------------------------


@jax.jit
def invsqrtm_supp(A: Array, eps: float = 1e-6) -> Array:
    r"""Support-restricted matrix inverse square root :math:`A^{-1/2}|_{\mathrm{supp}}`.

    Eigenvalues below ``eps`` are treated as zero and projected out, so
    rank-deficient :math:`A` produces a Moore--Penrose-style pseudoinverse
    on its support rather than blowing up to ``inf`` / ``nan``. For
    full-rank positive-definite :math:`A` with all eigenvalues above
    ``eps``, this agrees with the plain inverse square root.

    Parameters
    ----------
    A : Array, shape ``(N, N)``
        Hermitian (assumed positive-semidefinite) matrix.
    eps : float
        Eigenvalue support cutoff.

    Returns
    -------
    Array, shape ``(N, N)``
    """
    w, V = jnp.linalg.eigh(A)
    w_real = jnp.real(w)
    w_pos = jnp.maximum(w_real, eps)
    s = jnp.where(w_real > eps, 1.0 / jnp.sqrt(w_pos), 0.0)
    return V @ (s[:, None] * dag(V))


def sparse_eigh(O: Array, eps: float = 1e-6) -> tuple[Array, Array]:
    """Eigendecomposition keeping only eigenvalues >= *eps*.

    Parameters
    ----------
    O : Array, shape ``(N, N)``
        Hermitian matrix.
    eps : float
        Eigenvalue threshold.

    Returns
    -------
    eigenvalues : Array, shape ``(K,)``
    eigenvectors : Array, shape ``(N, K)``
    """
    lambda_O, U_O = jnp.linalg.eigh(O)
    mask = lambda_O >= eps
    return lambda_O[mask], U_O[:, mask]


def sparse_tensor_eigh(T: Array, eps: float = 1e-6) -> tuple[Array, Array]:
    """Eigendecomposition of a rank-4 block-Hermitian tensor.

    Reshapes ``T`` of shape ``(A, A, A, A)`` into ``(A^2, A^2)`` before
    calling :func:`sparse_eigh`.

    Returns
    -------
    eigenvalues : Array, shape ``(K,)``
    eigenmodes : Array, shape ``(A, A, K)``
    """
    A = T.shape[0]
    M = jnp.reshape(T, (A * A, A * A))
    w, U = sparse_eigh(M, eps=eps)
    chis = jnp.reshape(U, (A, A, w.shape[0]))
    return w, chis
