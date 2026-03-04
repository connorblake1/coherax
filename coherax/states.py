"""Coherent-basis quantum state representations.

Provides :class:`CoherentKet`, :class:`CoherentDM`, and
:class:`BosonicSubspace` for representing and manipulating bosonic code
states as superpositions of coherent states.
"""

from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from coherax.operators import (
    GKP_N,
    aOmegab,
    coherent_overlap,
    dag,
    dqcoherent,
    sparse_eigh,
)


class CoherentKet(eqx.Module):
    r"""Normalized coherent-state superposition :math:`|\psi\rangle = \sum_i c_i |d_i\rangle`.

    The coefficients are automatically normalized on construction so that
    :math:`\langle\psi|\psi\rangle = 1`.

    Parameters
    ----------
    cs : Array, shape ``(A,)``
        Coefficients of the superposition.
    ds : Array, shape ``(A,)``
        Complex displacement amplitudes of each coherent state.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from coherax.states import CoherentKet
    >>> cat = CoherentKet(cs=jnp.array([1.0, 1.0]),
    ...                   ds=jnp.array([2.0, -2.0]))
    """

    ds: Array
    cs: Array

    def __init__(self, cs: Array, ds: Array) -> None:
        ca = cs.reshape(-1, 1)
        cb = cs.reshape(1, -1)
        da = ds.reshape(-1, 1)
        db = ds.reshape(1, -1)
        phase = jnp.exp(1j * aOmegab(db, da))
        prefactor = ca * jnp.conj(cb)
        envelope = jnp.exp(-0.5 * jnp.abs(da - db) ** 2)
        norm = jnp.sqrt(jnp.real(jnp.sum(phase * prefactor * envelope)))
        self.ds = ds
        self.cs = cs / norm

    @jax.jit
    def __call__(self, u: complex) -> Array:
        r"""Evaluate the characteristic function :math:`\chi(u)`.

        Parameters
        ----------
        u : complex
            Phase-space point.

        Returns
        -------
        Array
            Scalar complex value of the characteristic function at *u*.
        """
        N = self.cs.shape[0]
        ca = self.cs.reshape(1, N)
        da = self.ds.reshape(1, N)
        cb = self.cs.reshape(N, 1)
        db = self.ds.reshape(N, 1)
        envelope = jnp.exp(
            -0.5 * jnp.abs(db - da - u) ** 2
            + 1j * (aOmegab(da, db) + aOmegab(u, da + db))
        )
        return jnp.sum(jnp.conj(ca) * cb * envelope)

    @jax.jit
    def to_fock_basis(self, N: int = GKP_N) -> Array:
        """Convert to a Fock-basis density matrix.

        Parameters
        ----------
        N : int
            Fock-space truncation dimension.

        Returns
        -------
        Array, shape ``(N, N)``
            Density matrix :math:`|\\psi\\rangle\\langle\\psi|` in the Fock basis.
        """
        coherents = jax.vmap(lambda alpha: dqcoherent(N, alpha))(self.ds)
        psi = jnp.einsum("ija,i->ja", coherents, self.cs).squeeze()
        return jnp.einsum("i,j->ij", psi, jnp.conj(psi))


class CoherentDM(eqx.Module):
    r"""Mixed state in the coherent basis: :math:`\rho = \sum_{ij} C_{ij} |d_i\rangle\langle d_j|`.

    Parameters
    ----------
    C : Array, shape ``(A, A)``
        Coefficient matrix.
    ds : Array, shape ``(A,)``
        Displacement amplitudes.
    """

    ds: Array
    C: Array

    def __init__(self, C: Array, ds: Array) -> None:
        G = coherent_overlap(ds.reshape((-1, 1)), ds.reshape((1, -1)))
        C = C / jnp.einsum("ij,ji", C, G)
        self.C = C
        self.ds = ds

    @partial(jax.jit, static_argnums=1)
    def to_fock_basis(self, N: int = GKP_N) -> Array:
        """Convert to a Fock-basis density matrix.

        Parameters
        ----------
        N : int
            Fock-space truncation.

        Returns
        -------
        Array, shape ``(N, N)``
        """
        coherents = jnp.squeeze(
            jax.vmap(lambda alpha: dqcoherent(N, alpha))(self.ds)
        )
        return jnp.einsum("ab,ai,bj->ij", self.C, coherents, jnp.conj(coherents))

    @staticmethod
    def from_ket(state: CoherentKet) -> CoherentDM:
        """Construct from a pure :class:`CoherentKet`."""
        return CoherentDM(
            C=jnp.einsum("i,j->ij", state.cs, jnp.conj(state.cs)), ds=state.ds
        )


class BosonicSubspace(eqx.Module):
    r"""Orthogonalized coherent-state subspace.

    Given a set of displacements :math:`\{d_i\}`, constructs the
    Gram matrix :math:`G_{ij} = \langle d_i | d_j \rangle` and
    change-of-basis matrices between the (overcomplete) coherent basis
    and an orthonormal basis obtained via eigendecomposition.

    Parameters
    ----------
    ds : Array, shape ``(A,)``
        Displacement amplitudes spanning the subspace.
    eps : float
        Eigenvalue cutoff for the Gram matrix.
    """

    ds: Array
    G: Array
    lambda_G: Array
    U_G: Array
    T: Array
    Tp: Array

    def __init__(self, ds: Array, eps: float = 1e-6) -> None:
        A = ds.shape[0]
        G = coherent_overlap(ds.reshape((A, 1)), ds.reshape((1, A)))
        lambda_G, U_G = sparse_eigh(G, eps)
        self.T = U_G @ jnp.diag(lambda_G**-0.5)
        self.Tp = jnp.diag(lambda_G**0.5) @ dag(U_G)
        self.ds = ds
        self.G = G
        self.lambda_G = lambda_G
        self.U_G = U_G

    def op_c2o_transform(self, O: Array) -> Array:
        """Transform an operator from the coherent basis to the orthonormal basis.

        Parameters
        ----------
        O : Array, shape ``(A, A)``

        Returns
        -------
        Array, shape ``(K, K)``
        """
        return jnp.einsum("ia,ab,jb->ij", self.Tp, O, jnp.conj(self.Tp))

    def op_o2c_transform(self, O: Array) -> Array:
        """Transform an operator from the orthonormal basis to the coherent basis."""
        return jnp.einsum("ai,ij,bj->ab", self.T, O, jnp.conj(self.T))

    def ket_c2o_transform(self, ket: Array) -> Array:
        """Transform a ket from the coherent basis to the orthonormal basis."""
        return jnp.einsum("ia,a->i", self.Tp, ket)

    def ket_o2c_transform(self, ket: Array) -> Array:
        """Transform a ket from the orthonormal basis to the coherent basis."""
        return jnp.einsum("ai,i->a", self.T, ket)

    def synthesize_ket_fock(self, ket: Array, N: int = GKP_N) -> Array:
        """Convert a coherent-basis ket to the Fock basis.

        Parameters
        ----------
        ket : Array, shape ``(A,)``
            Coefficients in the coherent basis.
        N : int
            Fock-space truncation.

        Returns
        -------
        Array, shape ``(N,)``
            Fock-basis ket.
        """
        coeffs = jnp.squeeze(
            jax.vmap(lambda alpha: dqcoherent(N, alpha))(self.ds)
        )
        return jnp.einsum("ai,a->i", coeffs, ket)

    @partial(jax.jit, static_argnums=2)
    def op_to_fock(self, O: Array, N: int = GKP_N) -> Array:
        """Convert a coherent-basis operator to the Fock basis.

        Parameters
        ----------
        O : Array, shape ``(A, A)``
            Operator in the coherent basis.
        N : int
            Fock-space truncation.

        Returns
        -------
        Array, shape ``(N, N)``
        """
        coherents = jnp.squeeze(
            jax.vmap(lambda alpha: dqcoherent(N, alpha))(self.ds)
        )
        return jnp.einsum("ai,bj,ab->ij", coherents, jnp.conj(coherents), O)
