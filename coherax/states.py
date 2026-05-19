"""Quantum state representations.

.. warning::
    This API is under active development and may change without notice.

Provides abstract :class:`Ket` and :class:`DM` base classes with concrete
implementations :class:`CoherentKet`, :class:`CoherentDM`, :class:`FockKet`,
:class:`FockDM`, :class:`QubitKet`, :class:`JointKet`, the typed
basis-defined operators :class:`CoherentCoherentOp`, :class:`FockFockOp`,
:class:`CoherentFockOp`, :class:`FockCoherentOp`, the analytic operators
:class:`Displacer`, :class:`Rotator`, :class:`CPTP`, and
:class:`BosonicSubspace`.
"""

from __future__ import annotations

from functools import partial
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from jaxtyping import Array

from coherax.linalg_utils import (
    GKP_N,
    aOmegab,
    coherent_overlap,
    dag,
    invsqrtm_supp,
)
from coherax._fock import dqcoherent


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------


class Ket(eqx.Module):
    r"""Abstract base class for pure quantum states :math:`|\psi\rangle`.

    Subclasses must implement :meth:`inner`, :meth:`to_fock_basis`,
    :meth:`to_fock_ket`, and :meth:`unit`.
    """

    def inner(self, other: Ket) -> Array:
        r"""Compute :math:`\langle\mathrm{self}|\mathrm{other}\rangle`.

        Parameters
        ----------
        other : Ket
            Right-hand ket.

        Returns
        -------
        Array
            Complex scalar inner product.
        """
        raise NotImplementedError

    def to_fock_basis(self, N: int = GKP_N) -> Array:
        r"""Convert to Fock-basis density matrix :math:`|\psi\rangle\langle\psi|`.

        Parameters
        ----------
        N : int
            Fock-space truncation dimension.

        Returns
        -------
        Array, shape ``(N, N)``
        """
        raise NotImplementedError

    def to_fock_ket(self, N: int = GKP_N) -> Array:
        """Convert to Fock-basis state vector.

        Parameters
        ----------
        N : int
            Fock-space truncation dimension.

        Returns
        -------
        Array, shape ``(N,)``
        """
        raise NotImplementedError

    def unit(self) -> Ket:
        r"""Return a normalized copy with :math:`\langle\psi|\psi\rangle = 1`.

        Returns
        -------
        Ket
            Normalized ket.
        """
        raise NotImplementedError


class DM(eqx.Module):
    r"""Abstract base class for density matrices :math:`\rho`.

    Subclasses must implement :meth:`inner`, :meth:`to_fock_basis`,
    and :meth:`unit`.
    """

    def inner(self, other: DM) -> Array:
        r"""Hilbert--Schmidt inner product :math:`\mathrm{Tr}(\rho\,\sigma^\dagger)`.

        Parameters
        ----------
        other : DM
            Right-hand density matrix.

        Returns
        -------
        Array
            Complex scalar.
        """
        raise NotImplementedError

    def to_fock_basis(self, N: int = GKP_N) -> Array:
        """Convert to Fock-basis density matrix.

        Parameters
        ----------
        N : int
            Fock-space truncation dimension.

        Returns
        -------
        Array, shape ``(N, N)``
        """
        raise NotImplementedError

    def unit(self) -> DM:
        r"""Return a trace-normalized copy with :math:`\mathrm{Tr}(\rho) = 1`.

        Returns
        -------
        DM
            Normalized density matrix.
        """
        raise NotImplementedError


def _abs_sq(z: Array) -> Array:
    r"""Compute :math:`|z|^2` without routing through ``sqrt``.

    ``jnp.abs(z)**2`` goes through ``sqrt`` internally, whose derivative
    diverges at zero and produces NaN gradients.  This computes
    ``re(z)^2 + im(z)^2`` directly so that gradients are always finite.

    Parameters
    ----------
    z : Array
        Complex (or real) array.

    Returns
    -------
    Array
        Real-valued :math:`|z|^2`.
    """
    return jnp.real(z * jnp.conj(z))


# ---------------------------------------------------------------------------
# Coherent-basis ket
# ---------------------------------------------------------------------------


class CoherentKet(Ket):
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
        cs = jnp.asarray(cs, dtype=jnp.complex128)
        ds = jnp.asarray(ds, dtype=jnp.complex128)
        ca = cs.reshape(-1, 1)
        cb = cs.reshape(1, -1)
        da = ds.reshape(-1, 1)
        db = ds.reshape(1, -1)
        phase = jnp.exp(1j * aOmegab(db, da))
        prefactor = ca * jnp.conj(cb)
        envelope = jnp.exp(-0.5 * _abs_sq(da - db))
        norm = jnp.sqrt(jnp.real(jnp.sum(phase * prefactor * envelope)))
        self.ds = ds
        self.cs = cs / norm

    def inner(self, other: Ket) -> Array:
        r"""Compute :math:`\langle\mathrm{self}|\mathrm{other}\rangle`.

        Supports :class:`CoherentKet` and :class:`FockKet` as *other*.

        **CoherentKet--CoherentKet** with
        :math:`|\psi\rangle = \sum_i c_i |d_i\rangle` and
        :math:`|\phi\rangle = \sum_j c'_j |d'_j\rangle`:

        .. math::

            \langle\psi|\phi\rangle
            = \sum_{i,j} c_i^*\, c'_j\,\langle d_i | d'_j \rangle,
            \quad
            \langle\alpha|\beta\rangle
            = e^{-\frac{1}{2}|\alpha - \beta|^2
              + i\,\operatorname{Im}(\alpha^*\beta)}.

        **CoherentKet--FockKet** with
        :math:`|\phi\rangle = \sum_a c'_a |n_a\rangle`:

        .. math::

            \langle\psi|\phi\rangle
            = \sum_{i,a} c_i^*\, c'_a\,\langle d_i | n_a \rangle,
            \quad
            \langle\alpha|n\rangle
            = e^{-|\alpha|^2/2}\,
              \frac{(\alpha^*)^n}{\sqrt{n!}}.

        Parameters
        ----------
        other : Ket
            A :class:`CoherentKet` or :class:`FockKet`.

        Returns
        -------
        Array
            Complex scalar inner product.
        """
        if isinstance(other, CoherentKet):
            overlap = coherent_overlap(self.ds.reshape(-1, 1), other.ds.reshape(1, -1))
            return jnp.sum(
                jnp.conj(self.cs).reshape(-1, 1) * other.cs.reshape(1, -1) * overlap
            )
        if isinstance(other, FockKet):
            return _inner_coherent_fock(self, other)
        raise TypeError(f"Cannot compute inner product with {type(other)}")

    @partial(jax.jit, static_argnums=1)
    def to_fock_ket(self, N: int = GKP_N) -> Array:
        """Convert to a Fock-basis state vector.

        Parameters
        ----------
        N : int
            Fock-space truncation dimension.

        Returns
        -------
        Array, shape ``(N,)``
        """
        coherents = jax.vmap(lambda alpha: dqcoherent(N, alpha))(self.ds)
        return jnp.einsum("ija,i->ja", coherents, self.cs).squeeze()

    @partial(jax.jit, static_argnums=1)
    def to_fock_basis(self, N: int = GKP_N) -> Array:
        r"""Convert to a Fock-basis density matrix.

        Parameters
        ----------
        N : int
            Fock-space truncation dimension.

        Returns
        -------
        Array, shape ``(N, N)``
            Density matrix :math:`|\psi\rangle\langle\psi|` in the Fock basis.
        """
        psi = self.to_fock_ket(N)
        return jnp.einsum("i,j->ij", psi, jnp.conj(psi))

    def unit(self) -> CoherentKet:
        r"""Return a normalized copy with :math:`\langle\psi|\psi\rangle = 1`.

        Returns
        -------
        CoherentKet
        """
        return CoherentKet(cs=self.cs, ds=self.ds)


# ---------------------------------------------------------------------------
# Fock-basis ket
# ---------------------------------------------------------------------------


class FockKet(Ket):
    r"""Pure state in a Fock subspace :math:`|\psi\rangle = \sum_{a=1}^A c_a |n_a\rangle`.

    The number states :math:`n_a` need not be contiguous (e.g. ``{0, 1, 5, 6}``
    is valid).  Coefficients are automatically normalized on construction.

    Parameters
    ----------
    cs : Array, shape ``(A,)``
        Coefficients of the superposition.
    ns : Array, shape ``(A,)`` | int
        Fock-state indices.  If an integer *k* is passed, interpreted as
        ``jnp.arange(k)`` (the first *k* Fock states).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from coherax.states import FockKet
    >>> psi = FockKet(cs=jnp.array([1.0, 0.0, 1.0]),
    ...              ns=jnp.array([0, 3, 7]))
    """

    cs: Array
    ns: Array

    def __init__(self, cs: Array, ns: Array | int) -> None:
        if isinstance(ns, int):
            ns = jnp.arange(ns)
        ns = jnp.asarray(ns)
        cs = jnp.asarray(cs, dtype=jnp.complex128)
        delta = (ns.reshape(-1, 1) == ns.reshape(1, -1)).astype(jnp.float64)
        norm_sq = jnp.real(jnp.einsum("i,j,ij->", jnp.conj(cs), cs, delta))
        self.cs = cs / jnp.sqrt(norm_sq)
        self.ns = ns

    def inner(self, other: Ket) -> Array:
        r"""Compute :math:`\langle\mathrm{self}|\mathrm{other}\rangle`.

        Supports :class:`FockKet` and :class:`CoherentKet` as *other*.

        **FockKet--FockKet** with
        :math:`|\psi\rangle = \sum_a c_a |n_a\rangle` and
        :math:`|\phi\rangle = \sum_b c'_b |n'_b\rangle`:

        .. math::

            \langle\psi|\phi\rangle
            = \sum_{a,b} c_a^*\, c'_b\,\delta_{n_a, n'_b}.

        **FockKet--CoherentKet** with
        :math:`|\phi\rangle = \sum_j c'_j |d'_j\rangle`:

        .. math::

            \langle\psi|\phi\rangle
            = \sum_{a,j} c_a^*\, c'_j\,\langle n_a | d'_j \rangle,
            \quad
            \langle n|\alpha\rangle
            = e^{-|\alpha|^2/2}\,
              \frac{\alpha^n}{\sqrt{n!}}.

        Parameters
        ----------
        other : Ket
            A :class:`FockKet` or :class:`CoherentKet`.

        Returns
        -------
        Array
            Complex scalar inner product.
        """
        if isinstance(other, FockKet):
            delta = (self.ns.reshape(-1, 1) == other.ns.reshape(1, -1)).astype(
                jnp.complex128
            )
            return jnp.sum(
                jnp.conj(self.cs).reshape(-1, 1) * other.cs.reshape(1, -1) * delta
            )
        if isinstance(other, CoherentKet):
            return jnp.conj(_inner_coherent_fock(other, self))
        raise TypeError(f"Cannot compute inner product with {type(other)}")

    def to_fock_ket(self, N: int = GKP_N) -> Array:
        """Convert to a Fock-basis state vector.

        Parameters
        ----------
        N : int
            Fock-space truncation dimension.

        Returns
        -------
        Array, shape ``(N,)``
        """
        psi = jnp.zeros(N, dtype=jnp.complex128)
        return psi.at[self.ns].add(self.cs)

    def to_fock_basis(self, N: int = GKP_N) -> Array:
        r"""Convert to a Fock-basis density matrix :math:`|\psi\rangle\langle\psi|`.

        Parameters
        ----------
        N : int
            Fock-space truncation dimension.

        Returns
        -------
        Array, shape ``(N, N)``
        """
        psi = self.to_fock_ket(N)
        return jnp.outer(psi, jnp.conj(psi))

    def unit(self) -> FockKet:
        r"""Return a normalized copy with :math:`\langle\psi|\psi\rangle = 1`.

        Returns
        -------
        FockKet
        """
        return FockKet(cs=self.cs, ns=self.ns)


# ---------------------------------------------------------------------------
# Qubit ket
# ---------------------------------------------------------------------------


class QubitKet(FockKet):
    r"""Two-level quantum state :math:`|\psi\rangle = c_0|0\rangle + c_1|1\rangle`.

    Convenience subclass of :class:`FockKet` with ``ns = [0, 1]``.

    Parameters
    ----------
    cs : Array, shape ``(2,)``
        Coefficients :math:`[c_0, c_1]`.
    """

    def __init__(self, cs: Array) -> None:
        super().__init__(cs=cs, ns=jnp.array([0, 1]))


class LogicalKet(FockKet):
    r"""State in a *D*-dimensional orthonormal logical subspace of Fock space.

    A :class:`LogicalKet` is a :class:`FockKet` whose ``ns`` indices are
    distinct (giving an automatically orthonormal subspace) and whose
    coefficient vector encodes the logical state
    :math:`|\psi\rangle = \sum_{k=0}^{D-1} c_k |n_k\rangle`. By default
    the logical basis is the first *D* Fock states (``ns = [0, 1, ..., D-1]``);
    pass ``ns`` explicitly to embed in a non-contiguous subspace. The
    coefficients are normalized on construction by :class:`FockKet`.

    Parameters
    ----------
    cs : Array, shape ``(D,)``
        Logical-basis coefficients.
    ns : Array, shape ``(D,)`` | None
        Fock indices spanning the logical subspace. Must be all distinct.
        Defaults to ``jnp.arange(D)``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from coherax.states import LogicalKet
    >>> # |0_L> + |2_L> in a 3-d subspace
    >>> psi = LogicalKet(cs=jnp.array([1.0, 0.0, 1.0]))
    >>> # 3-d subspace embedded at Fock levels 0, 2, 5
    >>> psi2 = LogicalKet(cs=jnp.array([1.0, 0.0, 1.0]), ns=jnp.array([0, 2, 5]))
    """

    def __init__(self, cs: Array, ns: Array | None = None) -> None:
        if ns is None:
            ns = jnp.arange(cs.shape[0])
        super().__init__(cs=cs, ns=ns)


# ---------------------------------------------------------------------------
# Joint ket (bosonic mode x qubit)
# ---------------------------------------------------------------------------


class JointKet(Ket):
    r"""Joint coherent-bosonic--qubit state.

    Represents the bipartite state

    .. math::

        |\Psi\rangle = \sum_{\mu=0}^{1}\sum_{a=1}^{A}
        c_{\mu a}\,|d_{\mu a}\rangle\otimes|\mu\rangle

    where the first register is a coherent-state superposition and the
    second register is a qubit.

    Parameters
    ----------
    cs : Array, shape ``(2, A)``
        Coefficients :math:`c_{\mu a}`.
    ds : Array, shape ``(2, A)``
        Complex displacement amplitudes :math:`d_{\mu a}`.

    Notes
    -----
    The state is automatically normalized on construction so that
    :math:`\langle\Psi|\Psi\rangle = 1`.  The normalization uses the
    coherent-state overlap:

    .. math::

        \langle\Psi|\Psi\rangle
        = \sum_{\mu}\sum_{a,b}
          c_{\mu a}^*\,c_{\mu b}\,\langle d_{\mu a}|d_{\mu b}\rangle
    """

    cs: Array
    ds: Array

    def __init__(self, cs: Array, ds: Array) -> None:
        cs = jnp.asarray(cs, dtype=jnp.complex128)
        ds = jnp.asarray(ds, dtype=jnp.complex128)
        # Normalize: <Psi|Psi> = sum_mu sum_{a,b} c*_{mu,a} c_{mu,b} <d_{mu,a}|d_{mu,b}>
        norm_sq = 0.0
        for mu in range(2):
            overlap = coherent_overlap(
                ds[mu].reshape(-1, 1), ds[mu].reshape(1, -1)
            )
            norm_sq = norm_sq + jnp.real(
                jnp.sum(
                    jnp.conj(cs[mu]).reshape(-1, 1)
                    * cs[mu].reshape(1, -1)
                    * overlap
                )
            )
        self.cs = cs / jnp.sqrt(norm_sq)
        self.ds = ds

    def inner(self, other: Ket) -> Array | Ket:
        r"""Inner product or partial inner product.

        Supports three cases:

        **JointKet--JointKet** (full inner product):

        .. math::

            \langle\Psi'|\Psi\rangle
            = \sum_{\mu}\sum_{a,b}
              c'^*_{\mu a}\,c_{\mu b}\,
              \langle d'_{\mu a}|d_{\mu b}\rangle

        **JointKet--QubitKet** (partial trace over qubit, returns
        :class:`CoherentKet`):

        .. math::

            \langle q|\Psi\rangle_{\mathrm{qubit}}
            = \sum_{\mu} q_\mu^*
              \sum_a c_{\mu a}\,|d_{\mu a}\rangle

        **JointKet--CoherentKet** (partial trace over bosonic mode,
        returns :class:`QubitKet`):

        .. math::

            \langle\varphi|\Psi\rangle_{\mathrm{boson}}
            = \sum_{\mu}
              \Bigl(\sum_{j,a} f_j^*\,c_{\mu a}\,
              \langle\alpha_j|d_{\mu a}\rangle\Bigr)\,|\mu\rangle

        Parameters
        ----------
        other : Ket
            A :class:`JointKet`, :class:`QubitKet`, or :class:`CoherentKet`.

        Returns
        -------
        Array or Ket
            Scalar for JointKet, :class:`CoherentKet` for QubitKet,
            :class:`QubitKet` for CoherentKet.
        """
        if isinstance(other, JointKet):
            result = jnp.array(0.0, dtype=jnp.complex128)
            for mu in range(2):
                overlap = coherent_overlap(
                    self.ds[mu].reshape(-1, 1), other.ds[mu].reshape(1, -1)
                )
                result = result + jnp.sum(
                    jnp.conj(self.cs[mu]).reshape(-1, 1)
                    * other.cs[mu].reshape(1, -1)
                    * overlap
                )
            return result
        if isinstance(other, QubitKet):
            # Partial inner product over qubit: <q|Psi>_qubit -> CoherentKet
            # Result: sum_mu conj(q_mu) * sum_a c_{mu,a} |d_{mu,a}>
            all_ds = jnp.concatenate([self.ds[0], self.ds[1]])
            all_cs = jnp.concatenate([
                jnp.conj(other.cs[0]) * self.cs[0],
                jnp.conj(other.cs[1]) * self.cs[1],
            ])
            return CoherentKet(cs=all_cs, ds=all_ds)
        if isinstance(other, CoherentKet):
            # Partial inner product over bosonic mode: <phi|Psi>_boson -> QubitKet
            # cs[mu] = sum_{j,a} conj(f_j) * c_{mu,a} * <alpha_j|d_{mu,a}>
            qubit_cs = jnp.zeros(2, dtype=jnp.complex128)
            for mu in range(2):
                overlap = coherent_overlap(
                    other.ds.reshape(-1, 1), self.ds[mu].reshape(1, -1)
                )
                qubit_cs = qubit_cs.at[mu].set(
                    jnp.sum(
                        jnp.conj(other.cs).reshape(-1, 1)
                        * self.cs[mu].reshape(1, -1)
                        * overlap
                    )
                )
            return QubitKet(cs=qubit_cs)
        raise TypeError(f"Cannot compute inner product with {type(other)}")

    @partial(jax.jit, static_argnums=1)
    def to_fock_ket(self, N: int = GKP_N) -> Array:
        r"""Convert to a Fock-basis state vector in the tensor product space.

        Uses the library-wide ``dqtensor(cavity, qubit) =
        kron(cavity, qubit)`` convention: cavity is the slow index and
        qubit is the fast index, so component ``[2n + mu]`` corresponds
        to cavity Fock state :math:`|n\rangle` and qubit state
        :math:`|\mu\rangle`. This matches the layout produced by
        :func:`coherax.circuits.CD`, :func:`coherax.circuits.ECD`, and
        :func:`coherax.circuits.circuit_layer`.

        .. math::

            |\Psi\rangle_{\mathrm{Fock}}
            = \sum_{\mu}
              \Bigl(\sum_a c_{\mu a}\,|d_{\mu a}\rangle\Bigr)
              \otimes |\mu\rangle
            \in \mathbb{C}^{2N}

        Parameters
        ----------
        N : int
            Fock-space truncation dimension for the bosonic mode.

        Returns
        -------
        Array, shape ``(2*N,)``
        """
        def cavity_ket(cs_mu: Array, ds_mu: Array) -> Array:
            coherents = jax.vmap(lambda alpha: dqcoherent(N, alpha))(ds_mu)
            return jnp.einsum("ija,i->ja", coherents, cs_mu).squeeze(-1)

        cav0 = cavity_ket(self.cs[0], self.ds[0])
        cav1 = cavity_ket(self.cs[1], self.ds[1])
        q0 = jnp.array([1.0 + 0j, 0.0 + 0j], dtype=jnp.complex128)
        q1 = jnp.array([0.0 + 0j, 1.0 + 0j], dtype=jnp.complex128)
        return jnp.kron(cav0, q0) + jnp.kron(cav1, q1)

    @partial(jax.jit, static_argnums=1)
    def to_fock_basis(self, N: int = GKP_N) -> Array:
        r"""Convert to a Fock-basis density matrix in the tensor product space.

        .. math::

            \rho = |\Psi\rangle\langle\Psi|
            \in \mathbb{C}^{2N \times 2N}

        Parameters
        ----------
        N : int
            Fock-space truncation dimension for the bosonic mode.

        Returns
        -------
        Array, shape ``(2*N, 2*N)``
        """
        psi = self.to_fock_ket(N)
        return jnp.outer(psi, jnp.conj(psi))

    def unit(self) -> JointKet:
        r"""Return a normalized copy with :math:`\langle\Psi|\Psi\rangle = 1`.

        Returns
        -------
        JointKet
        """
        return JointKet(cs=self.cs, ds=self.ds)


# ---------------------------------------------------------------------------
# Coherent-basis density matrix
# ---------------------------------------------------------------------------


class CoherentDM(DM):
    r"""Mixed state in the coherent basis: :math:`\rho = \sum_{ij} C_{ij} |d_i\rangle\langle d_j|`.

    Trace-normalized on construction.

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
        C = jnp.asarray(C, dtype=jnp.complex128)
        ds = jnp.asarray(ds, dtype=jnp.complex128)
        G = coherent_overlap(ds.reshape((-1, 1)), ds.reshape((1, -1)))
        C = C / jnp.einsum("ij,ji->", C, G)
        self.C = C
        self.ds = ds

    def inner(self, other: DM) -> Array:
        r"""Hilbert--Schmidt inner product :math:`\mathrm{Tr}(\rho\,\sigma^\dagger)`.

        Supports :class:`CoherentDM` and :class:`FockDM` as *other*.

        **CoherentDM--CoherentDM** with
        :math:`\rho = \sum_{ij} C_{ij}\,|d_i\rangle\langle d_j|` and
        :math:`\sigma = \sum_{kl} C'_{kl}\,|d'_k\rangle\langle d'_l|`:

        .. math::

            \mathrm{Tr}(\rho\,\sigma^\dagger)
            = \sum_{i,j,k,l}
              C_{ij}\,(C'_{kl})^*\,
              \langle d_j | d'_l \rangle\,
              \langle d'_k | d_i \rangle.

        **CoherentDM--FockDM** with
        :math:`\sigma = \sum_{kl} C'_{kl}\,|n_k\rangle\langle n_l|`:

        .. math::

            \mathrm{Tr}(\rho\,\sigma^\dagger)
            = \sum_{i,j,k,l}
              C_{ij}\,(C'_{kl})^*\,
              \langle d_j | n_l \rangle\,
              \langle n_k | d_i \rangle

        where the coherent--Fock overlaps are
        :math:`\langle\alpha|n\rangle = e^{-|\alpha|^2/2}\,(\alpha^*)^n / \sqrt{n!}`.

        Parameters
        ----------
        other : DM
            A :class:`CoherentDM` or :class:`FockDM`.

        Returns
        -------
        Array
            Complex scalar.
        """
        if isinstance(other, CoherentDM):
            G_jl = coherent_overlap(self.ds.reshape(-1, 1), other.ds.reshape(1, -1))
            G_ki = coherent_overlap(other.ds.reshape(-1, 1), self.ds.reshape(1, -1))
            return jnp.einsum("ij,kl,jl,ki->", self.C, jnp.conj(other.C), G_jl, G_ki)
        if isinstance(other, FockDM):
            return _dm_inner_coherent_fock(self, other)
        raise TypeError(f"Cannot compute inner product with {type(other)}")

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
        A = self.ds.shape[0]
        coherents = jax.vmap(lambda alpha: dqcoherent(N, alpha))(self.ds)
        coherents = coherents.reshape(A, N)
        return jnp.einsum("ab,ai,bj->ij", self.C, coherents, jnp.conj(coherents))

    def unit(self) -> CoherentDM:
        r"""Return a trace-normalized copy with :math:`\mathrm{Tr}(\rho) = 1`.

        Returns
        -------
        CoherentDM
        """
        return CoherentDM(C=self.C, ds=self.ds)

    @staticmethod
    def from_ket(state: CoherentKet) -> CoherentDM:
        """Construct from a pure :class:`CoherentKet`.

        Parameters
        ----------
        state : CoherentKet

        Returns
        -------
        CoherentDM
        """
        return CoherentDM(
            C=jnp.einsum("i,j->ij", state.cs, jnp.conj(state.cs)),
            ds=state.ds,
        )


# ---------------------------------------------------------------------------
# Fock-basis density matrix
# ---------------------------------------------------------------------------


class FockDM(DM):
    r"""Mixed state in a Fock subspace: :math:`\rho = \sum_{ij} C_{ij} |n_i\rangle\langle n_j|`.

    Trace-normalized on construction.

    Parameters
    ----------
    C : Array, shape ``(A, A)``
        Coefficient matrix.
    ns : Array, shape ``(A,)`` | int
        Fock-state indices.  If an integer *k* is passed, interpreted as
        ``jnp.arange(k)`` (the first *k* Fock states).
    """

    C: Array
    ns: Array

    def __init__(self, C: Array, ns: Array | int) -> None:
        if isinstance(ns, int):
            ns = jnp.arange(ns)
        ns = jnp.asarray(ns)
        C = jnp.asarray(C, dtype=jnp.complex128)
        delta = (ns.reshape(-1, 1) == ns.reshape(1, -1)).astype(jnp.float64)
        tr = jnp.real(jnp.einsum("ij,ji->", C, delta))
        self.C = C / tr
        self.ns = ns

    def inner(self, other: DM) -> Array:
        r"""Hilbert--Schmidt inner product :math:`\mathrm{Tr}(\rho\,\sigma^\dagger)`.

        Supports :class:`FockDM` and :class:`CoherentDM` as *other*.

        **FockDM--FockDM** with
        :math:`\rho = \sum_{ij} C_{ij}\,|n_i\rangle\langle n_j|` and
        :math:`\sigma = \sum_{kl} C'_{kl}\,|n'_k\rangle\langle n'_l|`:

        .. math::

            \mathrm{Tr}(\rho\,\sigma^\dagger)
            = \sum_{i,j,k,l}
              C_{ij}\,(C'_{kl})^*\,
              \delta_{n_j,\,n'_l}\,
              \delta_{n'_k,\,n_i}.

        **FockDM--CoherentDM** with
        :math:`\sigma = \sum_{kl} C'_{kl}\,|d'_k\rangle\langle d'_l|`:

        .. math::

            \mathrm{Tr}(\rho\,\sigma^\dagger)
            = \sum_{i,j,k,l}
              C_{ij}\,(C'_{kl})^*\,
              \langle n_j | d'_l \rangle\,
              \langle d'_k | n_i \rangle

        where :math:`\langle n|\alpha\rangle = e^{-|\alpha|^2/2}\,\alpha^n / \sqrt{n!}`.

        Parameters
        ----------
        other : DM
            A :class:`FockDM` or :class:`CoherentDM`.

        Returns
        -------
        Array
            Complex scalar.
        """
        if isinstance(other, FockDM):
            delta_jl = (self.ns.reshape(-1, 1) == other.ns.reshape(1, -1)).astype(
                jnp.complex128
            )
            delta_ki = (other.ns.reshape(-1, 1) == self.ns.reshape(1, -1)).astype(
                jnp.complex128
            )
            return jnp.einsum(
                "ij,kl,jl,ki->",
                self.C,
                jnp.conj(other.C),
                delta_jl,
                delta_ki,
            )
        if isinstance(other, CoherentDM):
            return _dm_inner_fock_coherent(self, other)
        raise TypeError(f"Cannot compute inner product with {type(other)}")

    def to_fock_basis(self, N: int = GKP_N) -> Array:
        """Convert to a Fock-basis density matrix.

        Parameters
        ----------
        N : int
            Fock-space truncation dimension.

        Returns
        -------
        Array, shape ``(N, N)``
        """
        A = self.ns.shape[0]
        idx_i = jnp.repeat(self.ns, A)
        idx_j = jnp.tile(self.ns, A)
        vals = self.C.ravel()
        rho = jnp.zeros((N, N), dtype=jnp.complex128)
        return rho.at[idx_i, idx_j].add(vals)

    def unit(self) -> FockDM:
        r"""Return a trace-normalized copy with :math:`\mathrm{Tr}(\rho) = 1`.

        Returns
        -------
        FockDM
        """
        return FockDM(C=self.C, ns=self.ns)

    @staticmethod
    def from_ket(state: FockKet) -> FockDM:
        """Construct from a pure :class:`FockKet`.

        Parameters
        ----------
        state : FockKet

        Returns
        -------
        FockDM
        """
        return FockDM(
            C=jnp.einsum("i,j->ij", state.cs, jnp.conj(state.cs)),
            ns=state.ns,
        )


# ---------------------------------------------------------------------------
# Private inner-product helpers
# ---------------------------------------------------------------------------


def _safe_complex_power(z: Array, n: Array) -> Array:
    r"""Compute :math:`z^n` without NaN gradients at :math:`z = 0`.

    JAX's default ``z ** n`` uses ``exp(n * log(z))`` which gives NaN
    gradients when *z* = 0.  This implementation multiplies via
    ``abs(z)^n * exp(i*n*angle(z))`` and uses ``jnp.where`` to mask
    the zero case, keeping both the forward and backward passes clean.

    Parameters
    ----------
    z : Array
        Complex base (broadcastable).
    n : Array
        Real exponent (broadcastable).

    Returns
    -------
    Array
        Complex result.
    """
    r_sq = _abs_sq(z)
    is_zero = r_sq == 0.0
    # Guard inputs to log/angle so that gradients stay finite even on
    # the masked branch (JAX evaluates both branches of ``where``).
    safe_z = jnp.where(is_zero, 1.0 + 0j, z)
    safe_r = jnp.sqrt(jnp.where(is_zero, 1.0, r_sq))
    mag = jnp.exp(n * jnp.log(safe_r))
    phase = jnp.exp(1j * n * jnp.angle(safe_z))
    return jnp.where(is_zero, jnp.where(n == 0.0, 1.0 + 0j, 0.0 + 0j), mag * phase)


def _coherent_fock_braket(alpha: Array, n: Array) -> Array:
    r"""Compute :math:`\langle\alpha|n\rangle` element-wise.

    .. math::
        \langle\alpha|n\rangle = e^{-|\alpha|^2/2}\,
        \frac{(\alpha^*)^n}{\sqrt{n!}}

    Gradient-safe at :math:`\alpha = 0`.

    Parameters
    ----------
    alpha : Array
        Complex displacement amplitudes (broadcastable).
    n : Array
        Fock-state indices as floats (broadcastable).

    Returns
    -------
    Array
        Complex overlaps.
    """
    n_float = jnp.asarray(n, dtype=jnp.float64)
    return (
        jnp.exp(-0.5 * _abs_sq(alpha))
        * _safe_complex_power(jnp.conj(alpha), n_float)
        / jnp.exp(0.5 * jsp.gammaln(n_float + 1.0))
    )


def _inner_coherent_fock(coh: CoherentKet, fock: FockKet) -> Array:
    r"""Compute :math:`\langle\psi_\mathrm{coh}|\phi_\mathrm{fock}\rangle`.

    With :math:`|\psi\rangle = \sum_i c_i |d_i\rangle` and
    :math:`|\phi\rangle = \sum_a c'_a |n_a\rangle`:

    .. math::

        \langle\psi|\phi\rangle
        = \sum_{i,a} c_i^*\, c'_a\,
          e^{-|d_i|^2/2}\,
          \frac{(d_i^*)^{n_a}}{\sqrt{n_a!}}.

    Parameters
    ----------
    coh : CoherentKet
        Bra state.
    fock : FockKet
        Ket state.

    Returns
    -------
    Array
        Complex scalar.
    """
    overlap = _coherent_fock_braket(
        coh.ds.reshape(-1, 1),
        fock.ns.reshape(1, -1),
    )
    return jnp.sum(jnp.conj(coh.cs).reshape(-1, 1) * fock.cs.reshape(1, -1) * overlap)


def _dm_inner_coherent_fock(cdm: CoherentDM, fdm: FockDM) -> Array:
    r"""Compute :math:`\mathrm{Tr}(\rho_\mathrm{coh}\,\sigma_\mathrm{fock}^\dagger)`.

    With :math:`\rho = \sum_{ij} C_{ij}\,|d_i\rangle\langle d_j|` and
    :math:`\sigma = \sum_{kl} C'_{kl}\,|n_k\rangle\langle n_l|`:

    .. math::

        \mathrm{Tr}(\rho\,\sigma^\dagger)
        = \sum_{i,j,k,l}
          C_{ij}\,(C'_{kl})^*\,
          \langle d_j | n_l \rangle\,
          \langle n_k | d_i \rangle.

    Parameters
    ----------
    cdm : CoherentDM
    fdm : FockDM

    Returns
    -------
    Array
        Complex scalar.
    """
    ns_float = fdm.ns.astype(jnp.float64)
    # <d_j|n_l>  shape (A_c, A_f)
    G_jl = _coherent_fock_braket(cdm.ds.reshape(-1, 1), ns_float.reshape(1, -1))
    # <n_k|d_i> = conj(<d_i|n_k>)  shape (A_f, A_c)
    G_ki = jnp.conj(
        _coherent_fock_braket(cdm.ds.reshape(1, -1), ns_float.reshape(-1, 1))
    )
    return jnp.einsum("ij,kl,jl,ki->", cdm.C, jnp.conj(fdm.C), G_jl, G_ki)


def _dm_inner_fock_coherent(fdm: FockDM, cdm: CoherentDM) -> Array:
    r"""Compute :math:`\mathrm{Tr}(\rho_\mathrm{fock}\,\sigma_\mathrm{coh}^\dagger)`.

    With :math:`\rho = \sum_{ij} C_{ij}\,|n_i\rangle\langle n_j|` and
    :math:`\sigma = \sum_{kl} C'_{kl}\,|d'_k\rangle\langle d'_l|`:

    .. math::

        \mathrm{Tr}(\rho\,\sigma^\dagger)
        = \sum_{i,j,k,l}
          C_{ij}\,(C'_{kl})^*\,
          \langle n_j | d'_l \rangle\,
          \langle d'_k | n_i \rangle.

    Parameters
    ----------
    fdm : FockDM
    cdm : CoherentDM

    Returns
    -------
    Array
        Complex scalar.
    """
    ns_float = fdm.ns.astype(jnp.float64)
    # <n_j|d_l> = conj(<d_l|n_j>)  shape (A_f, A_c)
    G_jl = jnp.conj(
        _coherent_fock_braket(cdm.ds.reshape(1, -1), ns_float.reshape(-1, 1))
    )
    # <d_k|n_i>  shape (A_c, A_f)
    G_ki = _coherent_fock_braket(cdm.ds.reshape(-1, 1), ns_float.reshape(1, -1))
    return jnp.einsum("ij,kl,jl,ki->", fdm.C, jnp.conj(cdm.C), G_jl, G_ki)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Typed basis-defined operators
#
# Each class represents O = sum_i |phi_i><psi_i| with one specific basis
# combination on the from- and to-sides. Domain and codomain kets are stored
# as stacked (M, A) arrays so that apply/wrap are pure einsum kernels --
# fully jittable, no Python-level branching, no list iteration. M is the
# number of basis kets; A is the (fixed) number of coherent/Fock terms per
# basis ket. To use kets with different A, pad with zero-weight terms.
#
# All four classes share a uniform interface:
#   apply(psi)      -> matching-type output ket
#   apply_adj(psi)  -> matching-type output ket (acts as O^dagger)
#   dagger()        -> swapped operator type
#   wrap(rho)       -> O rho O^dagger as a matching-type DM
#
# Output kets/DMs are normalized by their constructors, matching the
# convention of the rest of the library. Use these classes when you want
# to stay within a fixed basis pair; convert explicitly when you need to
# cross bases.
# ---------------------------------------------------------------------------


def _coherent_overlap_batched(ds_a: Array, ds_b: Array) -> Array:
    r"""Pairwise :math:`\langle d_a | d_b \rangle` for stacked displacements.

    Parameters
    ----------
    ds_a : Array, shape ``(..., A)``
    ds_b : Array, shape ``(..., B)``

    Returns
    -------
    Array, shape broadcast of ``(..., A, B)``
    """
    return coherent_overlap(ds_a[..., :, None], ds_b[..., None, :])


def _fock_delta_batched(ns_a: Array, ns_b: Array) -> Array:
    r"""Pairwise :math:`\delta_{n_a, n_b}` for stacked Fock indices."""
    return (ns_a[..., :, None] == ns_b[..., None, :]).astype(jnp.complex128)


class CoherentCoherentOp(eqx.Module):
    r"""Operator :math:`O = \sum_i |\phi_i\rangle\langle\psi_i|` with coherent-basis kets on both sides.

    Each domain ket :math:`|\psi_i\rangle = \sum_a \mathrm{cs\_from}[i,a]\,
    |\mathrm{ds\_from}[i,a]\rangle` and similarly for codomain kets
    :math:`|\phi_i\rangle`. All M basis kets must share the same per-ket
    term count (``A_from`` and ``A_to``); pad with zero coefficients if
    needed.

    Parameters
    ----------
    cs_from, ds_from : Array, shape ``(M, A_from)``
        Coefficient and displacement stacks for the domain basis kets.
    cs_to, ds_to : Array, shape ``(M, A_to)``
        Coefficient and displacement stacks for the codomain basis kets.
    """

    cs_from: Array
    ds_from: Array
    cs_to: Array
    ds_to: Array

    def __init__(self, cs_from: Array, ds_from: Array, cs_to: Array, ds_to: Array) -> None:
        cs_from = jnp.asarray(cs_from, dtype=jnp.complex128)
        ds_from = jnp.asarray(ds_from, dtype=jnp.complex128)
        cs_to = jnp.asarray(cs_to, dtype=jnp.complex128)
        ds_to = jnp.asarray(ds_to, dtype=jnp.complex128)
        if cs_from.shape != ds_from.shape:
            raise ValueError(f"cs_from {cs_from.shape} != ds_from {ds_from.shape}")
        if cs_to.shape != ds_to.shape:
            raise ValueError(f"cs_to {cs_to.shape} != ds_to {ds_to.shape}")
        if cs_from.ndim != 2 or cs_to.ndim != 2:
            raise ValueError("cs_from and cs_to must be 2D (M, A) arrays")
        if cs_from.shape[0] != cs_to.shape[0]:
            raise ValueError(
                f"cs_from has M={cs_from.shape[0]} kets but cs_to has M={cs_to.shape[0]}"
            )
        self.cs_from = cs_from
        self.ds_from = ds_from
        self.cs_to = cs_to
        self.ds_to = ds_to

    @classmethod
    def from_kets(
        cls,
        kets_from: list[CoherentKet],
        kets_to: list[CoherentKet],
    ) -> CoherentCoherentOp:
        """Construct from lists of homogeneous :class:`CoherentKet` objects."""
        if len(kets_from) != len(kets_to):
            raise ValueError("kets_from and kets_to must have the same length")
        cs_from = jnp.stack([k.cs for k in kets_from])
        ds_from = jnp.stack([k.ds for k in kets_from])
        cs_to = jnp.stack([k.cs for k in kets_to])
        ds_to = jnp.stack([k.ds for k in kets_to])
        return cls(cs_from=cs_from, ds_from=ds_from, cs_to=cs_to, ds_to=ds_to)

    def apply(self, psi: CoherentKet) -> CoherentKet:
        r"""Apply :math:`O|\psi\rangle = \sum_i \langle\psi_i|\psi\rangle\,|\phi_i\rangle`.

        .. math::

            \langle\psi_i|\psi\rangle
            = \sum_{a,b} \overline{\mathrm{cs\_from}[i,a]}\,
              \mathrm{cs}_\psi[b]\,
              \langle \mathrm{ds\_from}[i,a] | \mathrm{ds}_\psi[b] \rangle
        """
        G = _coherent_overlap_batched(self.ds_from, psi.ds[None, :])  # (M, A_from, A_psi)
        weights = jnp.einsum("ia,b,iab->i", jnp.conj(self.cs_from), psi.cs, G)
        out_cs = (weights[:, None] * self.cs_to).reshape(-1)
        out_ds = self.ds_to.reshape(-1)
        return CoherentKet(cs=out_cs, ds=out_ds)

    def apply_adj(self, psi: CoherentKet) -> CoherentKet:
        r"""Apply :math:`O^\dagger|\psi\rangle = \sum_i \langle\phi_i|\psi\rangle\,|\psi_i\rangle`."""
        G = _coherent_overlap_batched(self.ds_to, psi.ds[None, :])
        weights = jnp.einsum("ia,b,iab->i", jnp.conj(self.cs_to), psi.cs, G)
        out_cs = (weights[:, None] * self.cs_from).reshape(-1)
        out_ds = self.ds_from.reshape(-1)
        return CoherentKet(cs=out_cs, ds=out_ds)

    def dagger(self) -> CoherentCoherentOp:
        return CoherentCoherentOp(
            cs_from=self.cs_to, ds_from=self.ds_to,
            cs_to=self.cs_from, ds_to=self.ds_from,
        )

    def wrap(self, rho: CoherentDM) -> CoherentDM:
        r"""Apply :math:`O \rho O^\dagger`.

        With :math:`W_{ij} = \langle\psi_i|\rho|\psi_j\rangle`, the output
        DM has matrix elements

        .. math::

            \rho^{\mathrm{out}}_{(i,b),(j,b')}
            = W_{ij}\,\mathrm{cs\_to}[i,b]\,
              \overline{\mathrm{cs\_to}[j,b']}

        in the basis of displacements :math:`\mathrm{ds\_to}[i,b]`.
        """
        # H[i, p] = sum_a conj(cs_from[i, a]) * <ds_from[i, a] | rho.ds[p]>
        G = _coherent_overlap_batched(self.ds_from, rho.ds[None, :])  # (M, A_from, A_rho)
        H = jnp.einsum("ia,iap->ip", jnp.conj(self.cs_from), G)
        W = H @ rho.C @ dag(H)  # (M, M)
        # C^out[i, b, j, b'] = W[i, j] * cs_to[i, b] * conj(cs_to[j, b'])
        C_out_4d = (
            W[:, None, :, None]
            * self.cs_to[:, :, None, None]
            * jnp.conj(self.cs_to)[None, None, :, :]
        )
        M, A_to = self.cs_to.shape
        C_out = C_out_4d.reshape(M * A_to, M * A_to)
        ds_out = self.ds_to.reshape(-1)
        return CoherentDM(C=C_out, ds=ds_out)


class FockFockOp(eqx.Module):
    r"""Operator :math:`O = \sum_i |\phi_i\rangle\langle\psi_i|` with Fock-basis kets on both sides.

    Each :math:`|\psi_i\rangle = \sum_a \mathrm{cs\_from}[i,a]\,
    |\mathrm{ns\_from}[i,a]\rangle` and similarly for codomain.

    Parameters
    ----------
    cs_from : Array, shape ``(M, A_from)``
    ns_from : Array, shape ``(M, A_from)`` (integer)
    cs_to : Array, shape ``(M, A_to)``
    ns_to : Array, shape ``(M, A_to)`` (integer)
    """

    cs_from: Array
    ns_from: Array
    cs_to: Array
    ns_to: Array

    def __init__(self, cs_from: Array, ns_from: Array, cs_to: Array, ns_to: Array) -> None:
        cs_from = jnp.asarray(cs_from, dtype=jnp.complex128)
        cs_to = jnp.asarray(cs_to, dtype=jnp.complex128)
        ns_from = jnp.asarray(ns_from)
        ns_to = jnp.asarray(ns_to)
        if cs_from.shape != ns_from.shape:
            raise ValueError(f"cs_from {cs_from.shape} != ns_from {ns_from.shape}")
        if cs_to.shape != ns_to.shape:
            raise ValueError(f"cs_to {cs_to.shape} != ns_to {ns_to.shape}")
        if cs_from.ndim != 2 or cs_to.ndim != 2:
            raise ValueError("cs_from and cs_to must be 2D (M, A) arrays")
        if cs_from.shape[0] != cs_to.shape[0]:
            raise ValueError(
                f"cs_from has M={cs_from.shape[0]} kets but cs_to has M={cs_to.shape[0]}"
            )
        self.cs_from = cs_from
        self.ns_from = ns_from
        self.cs_to = cs_to
        self.ns_to = ns_to

    @classmethod
    def from_kets(
        cls,
        kets_from: list[FockKet],
        kets_to: list[FockKet],
    ) -> FockFockOp:
        """Construct from lists of homogeneous :class:`FockKet` objects."""
        if len(kets_from) != len(kets_to):
            raise ValueError("kets_from and kets_to must have the same length")
        cs_from = jnp.stack([k.cs for k in kets_from])
        ns_from = jnp.stack([k.ns for k in kets_from])
        cs_to = jnp.stack([k.cs for k in kets_to])
        ns_to = jnp.stack([k.ns for k in kets_to])
        return cls(cs_from=cs_from, ns_from=ns_from, cs_to=cs_to, ns_to=ns_to)

    def apply(self, psi: FockKet) -> FockKet:
        delta = _fock_delta_batched(self.ns_from, psi.ns[None, :])  # (M, A_from, A_psi)
        weights = jnp.einsum("ia,b,iab->i", jnp.conj(self.cs_from), psi.cs, delta)
        out_cs = (weights[:, None] * self.cs_to).reshape(-1)
        out_ns = self.ns_to.reshape(-1)
        return FockKet(cs=out_cs, ns=out_ns)

    def apply_adj(self, psi: FockKet) -> FockKet:
        delta = _fock_delta_batched(self.ns_to, psi.ns[None, :])
        weights = jnp.einsum("ia,b,iab->i", jnp.conj(self.cs_to), psi.cs, delta)
        out_cs = (weights[:, None] * self.cs_from).reshape(-1)
        out_ns = self.ns_from.reshape(-1)
        return FockKet(cs=out_cs, ns=out_ns)

    def dagger(self) -> FockFockOp:
        return FockFockOp(
            cs_from=self.cs_to, ns_from=self.ns_to,
            cs_to=self.cs_from, ns_to=self.ns_from,
        )

    def wrap(self, rho: FockDM) -> FockDM:
        # H[i, p] = sum_a conj(cs_from[i, a]) * delta(ns_from[i, a], rho.ns[p])
        delta = _fock_delta_batched(self.ns_from, rho.ns[None, :])
        H = jnp.einsum("ia,iap->ip", jnp.conj(self.cs_from), delta)
        W = H @ rho.C @ dag(H)
        C_out_4d = (
            W[:, None, :, None]
            * self.cs_to[:, :, None, None]
            * jnp.conj(self.cs_to)[None, None, :, :]
        )
        M, A_to = self.cs_to.shape
        return FockDM(C=C_out_4d.reshape(M * A_to, M * A_to), ns=self.ns_to.reshape(-1))


class CoherentFockOp(eqx.Module):
    r"""Operator with coherent-basis domain and Fock-basis codomain.

    :math:`O = \sum_i |\phi_i^{\mathrm{fock}}\rangle\langle\psi_i^{\mathrm{coh}}|`.

    Parameters
    ----------
    cs_from, ds_from : Array, shape ``(M, A_from)``
        Coherent-basis domain ket stacks.
    cs_to : Array, shape ``(M, A_to)``
    ns_to : Array, shape ``(M, A_to)`` (integer)
        Fock-basis codomain ket stacks.
    """

    cs_from: Array
    ds_from: Array
    cs_to: Array
    ns_to: Array

    def __init__(self, cs_from: Array, ds_from: Array, cs_to: Array, ns_to: Array) -> None:
        cs_from = jnp.asarray(cs_from, dtype=jnp.complex128)
        ds_from = jnp.asarray(ds_from, dtype=jnp.complex128)
        cs_to = jnp.asarray(cs_to, dtype=jnp.complex128)
        ns_to = jnp.asarray(ns_to)
        if cs_from.shape != ds_from.shape:
            raise ValueError(f"cs_from {cs_from.shape} != ds_from {ds_from.shape}")
        if cs_to.shape != ns_to.shape:
            raise ValueError(f"cs_to {cs_to.shape} != ns_to {ns_to.shape}")
        if cs_from.ndim != 2 or cs_to.ndim != 2:
            raise ValueError("cs_from and cs_to must be 2D (M, A) arrays")
        if cs_from.shape[0] != cs_to.shape[0]:
            raise ValueError(
                f"cs_from has M={cs_from.shape[0]} kets but cs_to has M={cs_to.shape[0]}"
            )
        self.cs_from = cs_from
        self.ds_from = ds_from
        self.cs_to = cs_to
        self.ns_to = ns_to

    @classmethod
    def from_kets(
        cls,
        kets_from: list[CoherentKet],
        kets_to: list[FockKet],
    ) -> CoherentFockOp:
        if len(kets_from) != len(kets_to):
            raise ValueError("kets_from and kets_to must have the same length")
        return cls(
            cs_from=jnp.stack([k.cs for k in kets_from]),
            ds_from=jnp.stack([k.ds for k in kets_from]),
            cs_to=jnp.stack([k.cs for k in kets_to]),
            ns_to=jnp.stack([k.ns for k in kets_to]),
        )

    def apply(self, psi: CoherentKet) -> FockKet:
        G = _coherent_overlap_batched(self.ds_from, psi.ds[None, :])
        weights = jnp.einsum("ia,b,iab->i", jnp.conj(self.cs_from), psi.cs, G)
        out_cs = (weights[:, None] * self.cs_to).reshape(-1)
        out_ns = self.ns_to.reshape(-1)
        return FockKet(cs=out_cs, ns=out_ns)

    def apply_adj(self, psi: FockKet) -> CoherentKet:
        # weights[i] = <phi_i^fock | psi^fock> = sum_a sum_b conj(cs_to[i,a])
        # * cs_psi[b] * delta(ns_to[i,a], ns_psi[b])
        delta = _fock_delta_batched(self.ns_to, psi.ns[None, :])
        weights = jnp.einsum("ia,b,iab->i", jnp.conj(self.cs_to), psi.cs, delta)
        out_cs = (weights[:, None] * self.cs_from).reshape(-1)
        out_ds = self.ds_from.reshape(-1)
        return CoherentKet(cs=out_cs, ds=out_ds)

    def dagger(self) -> FockCoherentOp:
        return FockCoherentOp(
            cs_from=self.cs_to, ns_from=self.ns_to,
            cs_to=self.cs_from, ds_to=self.ds_from,
        )

    def wrap(self, rho: CoherentDM) -> FockDM:
        # Domain is coherent (matches rho), codomain is fock.
        G = _coherent_overlap_batched(self.ds_from, rho.ds[None, :])
        H = jnp.einsum("ia,iap->ip", jnp.conj(self.cs_from), G)
        W = H @ rho.C @ dag(H)
        C_out_4d = (
            W[:, None, :, None]
            * self.cs_to[:, :, None, None]
            * jnp.conj(self.cs_to)[None, None, :, :]
        )
        M, A_to = self.cs_to.shape
        return FockDM(C=C_out_4d.reshape(M * A_to, M * A_to), ns=self.ns_to.reshape(-1))


class FockCoherentOp(eqx.Module):
    r"""Operator with Fock-basis domain and coherent-basis codomain.

    :math:`O = \sum_i |\phi_i^{\mathrm{coh}}\rangle\langle\psi_i^{\mathrm{fock}}|`.
    """

    cs_from: Array
    ns_from: Array
    cs_to: Array
    ds_to: Array

    def __init__(self, cs_from: Array, ns_from: Array, cs_to: Array, ds_to: Array) -> None:
        cs_from = jnp.asarray(cs_from, dtype=jnp.complex128)
        cs_to = jnp.asarray(cs_to, dtype=jnp.complex128)
        ds_to = jnp.asarray(ds_to, dtype=jnp.complex128)
        ns_from = jnp.asarray(ns_from)
        if cs_from.shape != ns_from.shape:
            raise ValueError(f"cs_from {cs_from.shape} != ns_from {ns_from.shape}")
        if cs_to.shape != ds_to.shape:
            raise ValueError(f"cs_to {cs_to.shape} != ds_to {ds_to.shape}")
        if cs_from.ndim != 2 or cs_to.ndim != 2:
            raise ValueError("cs_from and cs_to must be 2D (M, A) arrays")
        if cs_from.shape[0] != cs_to.shape[0]:
            raise ValueError(
                f"cs_from has M={cs_from.shape[0]} kets but cs_to has M={cs_to.shape[0]}"
            )
        self.cs_from = cs_from
        self.ns_from = ns_from
        self.cs_to = cs_to
        self.ds_to = ds_to

    @classmethod
    def from_kets(
        cls,
        kets_from: list[FockKet],
        kets_to: list[CoherentKet],
    ) -> FockCoherentOp:
        if len(kets_from) != len(kets_to):
            raise ValueError("kets_from and kets_to must have the same length")
        return cls(
            cs_from=jnp.stack([k.cs for k in kets_from]),
            ns_from=jnp.stack([k.ns for k in kets_from]),
            cs_to=jnp.stack([k.cs for k in kets_to]),
            ds_to=jnp.stack([k.ds for k in kets_to]),
        )

    def apply(self, psi: FockKet) -> CoherentKet:
        delta = _fock_delta_batched(self.ns_from, psi.ns[None, :])
        weights = jnp.einsum("ia,b,iab->i", jnp.conj(self.cs_from), psi.cs, delta)
        out_cs = (weights[:, None] * self.cs_to).reshape(-1)
        out_ds = self.ds_to.reshape(-1)
        return CoherentKet(cs=out_cs, ds=out_ds)

    def apply_adj(self, psi: CoherentKet) -> FockKet:
        G = _coherent_overlap_batched(self.ds_to, psi.ds[None, :])
        weights = jnp.einsum("ia,b,iab->i", jnp.conj(self.cs_to), psi.cs, G)
        out_cs = (weights[:, None] * self.cs_from).reshape(-1)
        out_ns = self.ns_from.reshape(-1)
        return FockKet(cs=out_cs, ns=out_ns)

    def dagger(self) -> CoherentFockOp:
        return CoherentFockOp(
            cs_from=self.cs_to, ds_from=self.ds_to,
            cs_to=self.cs_from, ns_to=self.ns_from,
        )

    def wrap(self, rho: FockDM) -> CoherentDM:
        # Domain is fock (matches rho), codomain is coherent.
        delta = _fock_delta_batched(self.ns_from, rho.ns[None, :])
        H = jnp.einsum("ia,iap->ip", jnp.conj(self.cs_from), delta)
        W = H @ rho.C @ dag(H)
        C_out_4d = (
            W[:, None, :, None]
            * self.cs_to[:, :, None, None]
            * jnp.conj(self.cs_to)[None, None, :, :]
        )
        M, A_to = self.cs_to.shape
        return CoherentDM(C=C_out_4d.reshape(M * A_to, M * A_to), ds=self.ds_to.reshape(-1))


class Displacer(eqx.Module):
    r"""Displacement operator :math:`D(\beta)`.

    Analytically applies to coherent states via the braiding relation:

    .. math::
        D(\beta)|\alpha\rangle = e^{i\,\operatorname{Im}(\beta\alpha^*)}\,|\alpha + \beta\rangle

    For a superposition :math:`|\psi\rangle = \sum_i c_i |d_i\rangle`:

    .. math::
        D(\beta)|\psi\rangle = \sum_i c_i\, e^{i\,\omega(d_i, \beta)}\,|d_i + \beta\rangle

    where :math:`\omega(a,b) = \operatorname{Re}(a)\operatorname{Im}(b) - \operatorname{Im}(a)\operatorname{Re}(b)`.

    Parameters
    ----------
    beta : complex or Array
        Displacement amplitude.
    """

    beta: Array

    def __init__(self, beta: complex | Array) -> None:
        self.beta = jnp.asarray(beta, dtype=jnp.complex128)

    def apply(self, psi: CoherentKet) -> CoherentKet:
        r"""Apply :math:`D(\beta)` to a coherent-state superposition.

        Parameters
        ----------
        psi : CoherentKet
            Input state.

        Returns
        -------
        CoherentKet
            Displaced state.
        """
        phase = jnp.exp(1j * aOmegab(psi.ds, self.beta))
        return CoherentKet(cs=psi.cs * phase, ds=psi.ds + self.beta)

    def apply_adj(self, psi: CoherentKet) -> CoherentKet:
        r"""Apply :math:`D^\dagger(\beta) = D(-\beta)`.

        Parameters
        ----------
        psi : CoherentKet
            Input state.

        Returns
        -------
        CoherentKet
            Displaced state.
        """
        phase = jnp.exp(1j * aOmegab(psi.ds, -self.beta))
        return CoherentKet(cs=psi.cs * phase, ds=psi.ds - self.beta)

    def apply_dm(self, rho: CoherentDM) -> CoherentDM:
        r"""Apply :math:`D(\beta)\,\rho\,D^\dagger(\beta)`.

        For :math:`\rho = \sum_{ij} C_{ij}\,|d_i\rangle\langle d_j|`:

        .. math::

            D(\beta)\,\rho\,D^\dagger(\beta)
            = \sum_{ij} C_{ij}\,
              e^{i[\omega(d_i,\beta) - \omega(d_j,\beta)]}\,
              |d_i + \beta\rangle\langle d_j + \beta|

        Parameters
        ----------
        rho : CoherentDM
            Input density matrix.

        Returns
        -------
        CoherentDM
            Displaced density matrix.
        """
        omega_i = aOmegab(rho.ds, self.beta)  # shape (A,)
        phase_ij = jnp.exp(
            1j * (omega_i.reshape(-1, 1) - omega_i.reshape(1, -1))
        )
        return CoherentDM(C=rho.C * phase_ij, ds=rho.ds + self.beta)

    def dagger(self) -> Displacer:
        r"""Return :math:`D^\dagger(\beta) = D(-\beta)`.

        Returns
        -------
        Displacer
        """
        return Displacer(beta=-self.beta)


class Rotator(eqx.Module):
    r"""Phase-space rotation :math:`e^{i\theta\hat{n}}`.

    Rotates coherent states:

    .. math::
        e^{i\theta\hat{n}}|\alpha\rangle = |\alpha e^{i\theta}\rangle

    Parameters
    ----------
    theta : float or Array
        Rotation angle.
    """

    theta: Array

    def __init__(self, theta: float | Array) -> None:
        self.theta = jnp.asarray(theta, dtype=jnp.float64)

    def apply(self, psi: CoherentKet) -> CoherentKet:
        r"""Apply the rotation to a coherent-state superposition.

        Parameters
        ----------
        psi : CoherentKet
            Input state.

        Returns
        -------
        CoherentKet
            Rotated state.
        """
        return CoherentKet(cs=psi.cs, ds=psi.ds * jnp.exp(1j * self.theta))

    def apply_adj(self, psi: CoherentKet) -> CoherentKet:
        r"""Apply the adjoint rotation :math:`e^{-i\theta\hat{n}}`.

        Parameters
        ----------
        psi : CoherentKet
            Input state.

        Returns
        -------
        CoherentKet
            Rotated state.
        """
        return CoherentKet(cs=psi.cs, ds=psi.ds * jnp.exp(-1j * self.theta))

    def apply_dm(self, rho: CoherentDM) -> CoherentDM:
        r"""Apply :math:`e^{i\theta\hat{n}}\,\rho\,e^{-i\theta\hat{n}}`.

        .. math::

            e^{i\theta\hat{n}}\,\rho\,e^{-i\theta\hat{n}}
            = \sum_{ij} C_{ij}\,
              |d_i e^{i\theta}\rangle\langle d_j e^{i\theta}|

        Parameters
        ----------
        rho : CoherentDM
            Input density matrix.

        Returns
        -------
        CoherentDM
            Rotated density matrix.
        """
        return CoherentDM(C=rho.C, ds=rho.ds * jnp.exp(1j * self.theta))

    def dagger(self) -> Rotator:
        r"""Return the adjoint rotation :math:`e^{-i\theta\hat{n}}`.

        Returns
        -------
        Rotator
        """
        return Rotator(theta=-self.theta)


class CPTP(eqx.Module):
    r"""Completely positive trace-preserving map (Kraus channel).

    .. math::
        \mathcal{E}(\rho) = \sum_k O_k \rho O_k^\dagger

    Stores a list of Kraus operators, each of which must have an
    ``apply`` method.  The :meth:`kraus_kets` method returns the
    individual Kraus branches :math:`O_k|\psi\rangle`, from which the
    output density matrix can be reconstructed as
    :math:`\rho_{\mathrm{out}} = \sum_k |\psi_k\rangle\langle\psi_k|`.

    Parameters
    ----------
    ops : list
        Kraus operators, each with an ``apply`` method.
    """

    ops: list

    def kraus_kets(self, psi: Ket) -> list[Ket]:
        r"""Apply each Kraus operator to a ket, returning the branches.

        .. math::
            \{O_k|\psi\rangle\}_{k=1}^{K}

        Parameters
        ----------
        psi : Ket
            Input pure state.

        Returns
        -------
        list[Ket]
            List of (unnormalized) output kets, one per Kraus operator.
        """
        return [op.apply(psi) for op in self.ops]


# ---------------------------------------------------------------------------
# Bosonic subspace (unchanged)
# ---------------------------------------------------------------------------


class BosonicSubspace(eqx.Module):
    r"""Orthogonalized coherent-state subspace.

    Given a set of displacements :math:`\{d_i\}`, constructs the
    Gram matrix :math:`G_{ij} = \langle d_i | d_j \rangle` and
    change-of-basis matrices between the (overcomplete) coherent basis
    and an orthonormal basis obtained from the eigendecomposition of
    :math:`G`. Eigenvalues at or below ``eps`` are masked out, so
    rank-deficient :math:`G` does not produce ``inf`` / ``nan`` --
    the corresponding columns of :attr:`T` and rows of :attr:`Tp` are
    zero instead. All shapes stay static at :math:`A`, so a
    :class:`BosonicSubspace` can be built inside ``jax.jit``.

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
        lambda_G, U_G = jnp.linalg.eigh(G)
        self.ds = ds
        self.G = G
        self.lambda_G = lambda_G
        self.U_G = U_G
        # Static-shape (A, A) versions of the original eigenvalue-basis
        # transforms: rows / columns corresponding to eigenvalues at or
        # below ``eps`` are zeroed out instead of being dropped. Active
        # rank is K = (lambda_G > eps).sum().
        w_real = jnp.real(lambda_G)
        inv_sqrt = jnp.where(w_real > eps, w_real ** -0.5, 0.0)
        sqrt_w = jnp.where(w_real > eps, w_real ** 0.5, 0.0)
        self.T = U_G * inv_sqrt[None, :]   # U_G @ diag(inv_sqrt), shape (A, A)
        self.Tp = sqrt_w[:, None] * dag(U_G)  # diag(sqrt) @ dag(U_G), shape (A, A)

    def op_c2o_transform(self, Op: Array) -> Array:
        """Transform an operator from the coherent basis to the orthonormal basis.

        Parameters
        ----------
        Op : Array, shape ``(A, A)``

        Returns
        -------
        Array, shape ``(A, A)``
            Orthonormal-basis matrix elements; components in the
            null-eigenvalue subspace of :math:`G` are zero.
        """
        return jnp.einsum("ia,ab,jb->ij", self.Tp, Op, jnp.conj(self.Tp))

    def op_o2c_transform(self, Op: Array) -> Array:
        """Transform an operator from the orthonormal basis to the coherent basis.

        Parameters
        ----------
        Op : Array, shape ``(A, A)``

        Returns
        -------
        Array, shape ``(A, A)``
        """
        return jnp.einsum("ai,ij,bj->ab", self.T, Op, jnp.conj(self.T))

    def ket_c2o_transform(self, ket: Array) -> Array:
        """Transform a ket from the coherent basis to the orthonormal basis.

        Low-level array method that preserves un-normalized magnitudes.
        For a typed entry point that wraps the result as a
        :class:`CoherentKet`, see :meth:`coherent_ket_to_orthonormal`.

        Parameters
        ----------
        ket : Array, shape ``(A,)``

        Returns
        -------
        Array, shape ``(A,)``
            Orthonormal-basis components; null-eigenvalue components are
            zero.
        """
        return jnp.einsum("ia,a->i", self.Tp, ket)

    def ket_o2c_transform(self, ket: Array) -> Array:
        """Transform a ket from the orthonormal basis to the coherent basis.

        Parameters
        ----------
        ket : Array, shape ``(A,)``

        Returns
        -------
        Array, shape ``(A,)``
        """
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
        coeffs = jnp.squeeze(jax.vmap(lambda alpha: dqcoherent(N, alpha))(self.ds))
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
        coherents = jnp.squeeze(jax.vmap(lambda alpha: dqcoherent(N, alpha))(self.ds))
        return jnp.einsum("ai,bj,ab->ij", coherents, jnp.conj(coherents), O)

    # ------------------------------------------------------------------
    # Typed CoherentKet entry points
    #
    # A "ket in this subspace" is just a :class:`CoherentKet` whose
    # displacements match ``self.ds``. These methods wrap the lower-level
    # array transforms so callers can work in the typed CoherentKet
    # framework. Note that :class:`CoherentKet` normalizes on
    # construction; use the underlying array methods if you need to keep
    # un-normalized intermediates.
    # ------------------------------------------------------------------

    def coherent_ket(self, cs: Array) -> CoherentKet:
        r"""Wrap a coefficient array as a :class:`CoherentKet` in this subspace.

        Parameters
        ----------
        cs : Array, shape ``(A,)``
            Coefficients in the coherent (non-orthonormal) basis with
            displacements ``self.ds``.

        Returns
        -------
        CoherentKet
            Normalized coherent-state superposition
            :math:`\sum_a c_a |d_a\rangle` with :math:`d_a = \mathrm{self.ds}[a]`.
        """
        return CoherentKet(cs=cs, ds=self.ds)

    def coherent_ket_to_orthonormal(self, ket: CoherentKet) -> Array:
        r"""Transform a :class:`CoherentKet`'s coefficients to the orthonormal basis.

        The caller is responsible for ensuring ``ket.ds == self.ds``; this
        method only consumes ``ket.cs``. Equivalent to
        ``self.ket_c2o_transform(ket.cs)``.

        Parameters
        ----------
        ket : CoherentKet
            A coherent ket whose displacements match ``self.ds``.

        Returns
        -------
        Array, shape ``(K,)``
            Coefficients in the orthonormal basis.
        """
        return self.ket_c2o_transform(ket.cs)

    def orthonormal_to_coherent_ket(self, coeffs: Array) -> CoherentKet:
        r"""Build a :class:`CoherentKet` from orthonormal-basis coefficients.

        Parameters
        ----------
        coeffs : Array, shape ``(K,)``
            Coefficients in the orthonormal basis.

        Returns
        -------
        CoherentKet
            Coherent-basis representation, normalized by
            :class:`CoherentKet` construction.
        """
        return CoherentKet(cs=self.ket_o2c_transform(coeffs), ds=self.ds)


# ---------------------------------------------------------------------------
# Beamsplitter channel on a logical encoding
#
# Given a logical encoding |psi_mu> = sum_a alpha_{mu, a} |beta_a> (a length-D
# stack of CoherentKets that share a single displacement vector `beta`), and
# an environment ket |env> = sum_j Y_j |sigma_j>, the beamsplitter with
# transmissivity eta acts as
#
#   U_BS (|beta_a> |sigma_j>) = |sqrt(eta) beta_a + sqrt(1-eta) sigma_j>_S
#                            otimes |sqrt(eta) sigma_j - sqrt(1-eta) beta_a>_E
#
# Tracing out the environment yields, for each (mu, nu) pair of logical
# states, a coherent-basis density-matrix block with displacements
# d_out_{a,j} = sqrt(eta) beta_a + sqrt(1-eta) sigma_j (shared across mu, nu)
# and coefficient matrix
#
#   C^{(mu nu)}_{(a,j),(b,k)} = alpha_{mu, a} conj(alpha_{nu, b})
#                                 * Y_j conj(Y_k)
#                                 * <d_mix_b,k | d_mix_a,j>
#
# where d_mix_{a,j} = sqrt(eta) sigma_j - sqrt(1-eta) beta_a is the
# environment-side displacement.
# ---------------------------------------------------------------------------


@jax.jit
def _beamsplit_full_arrays(
    alpha: Array,
    beta: Array,
    Y: Array,
    sigma: Array,
    eta: Array,
) -> tuple[Array, Array]:
    r"""Raw-array kernel for :func:`beamsplit_full`.

    All array indices are explicit so this function jit-cleanly. ``Y`` is
    normalized inside the kernel so callers do not have to pre-normalize
    the environment-mode coefficients.

    Parameters
    ----------
    alpha : Array, shape ``(D, A)``
        Coefficients of the encoded logical states in the (shared)
        coherent basis :math:`\{|\beta_a\rangle\}`.
    beta : Array, shape ``(A,)``
        Coherent-state displacements of the encoder output basis.
    Y : Array, shape ``(N_E,)``
        Environment-mode coefficients (pre-normalization).
    sigma : Array, shape ``(N_E,)``
        Environment-mode coherent displacements.
    eta : float or Array
        Beamsplitter transmissivity in :math:`[0, 1]`. Pure-loss
        channels use :math:`\eta = 1 - \gamma`.

    Returns
    -------
    rho_out : Array, shape ``(D, D, A * N_E, A * N_E)``
        Choi-like tensor: ``rho_out[mu, nu]`` is the unnormalized
        coherent-basis density-matrix block of
        :math:`\mathcal{E}(|\mu_L\rangle\!\langle\nu_L|)` with
        displacements ``d_out``.
    d_out : Array, shape ``(A * N_E,)``
        Output coherent-state displacements.
    """
    D, A = alpha.shape
    N_E = sigma.shape[0]
    beta_col = beta.reshape((A, 1))
    sigma_row = sigma.reshape((1, N_E))

    # Normalize Y in the environment's coherent overlap metric.
    G_env = coherent_overlap(sigma.reshape((N_E, 1)), sigma.reshape((1, N_E)))
    gamma = Y / jnp.sqrt(jnp.einsum("i,ij,j->", jnp.conj(Y), G_env, Y))

    d_out = jnp.sqrt(eta) * beta_col + jnp.sqrt(1.0 - eta) * sigma_row   # (A, N_E)
    d_mix = jnp.sqrt(eta) * sigma_row - jnp.sqrt(1.0 - eta) * beta_col   # (A, N_E)

    # Pairwise environment-side overlaps: G_mix[a, j, b, k] = <d_mix[b,k] | d_mix[a,j]>
    d_aj = d_mix.reshape((A, 1, N_E, 1))
    d_bk = d_mix.reshape((1, A, 1, N_E))
    G_mix = coherent_overlap(d_bk, d_aj)  # (A, A, N_E, N_E)

    rho_out = (
        G_mix.reshape((1, 1, A, A, N_E, N_E))
        * alpha.reshape((D, 1, A, 1, 1, 1))
        * jnp.conj(alpha).reshape((1, D, 1, A, 1, 1))
        * gamma.reshape((1, 1, 1, 1, N_E, 1))
        * jnp.conj(gamma).reshape((1, 1, 1, 1, 1, N_E))
    )
    # Reorder (D, D, A, A, N_E, N_E) -> (D, D, A*N_E, A*N_E)
    rho_out = jnp.transpose(rho_out, (0, 1, 2, 4, 3, 5)).reshape(
        D, D, A * N_E, A * N_E
    )
    return rho_out, d_out.reshape((A * N_E,))


def beamsplit_full(
    logical_kets: Sequence[CoherentKet],
    env: CoherentKet,
    eta: float | Array,
) -> tuple[Array, Array]:
    r"""Apply a beamsplitter to a logical encoding and trace out the environment.

    The encoder is described by a length-:math:`D` sequence of
    :class:`CoherentKet`\ s ``logical_kets[mu]`` :math:`= \sum_a
    \alpha_{\mu a}|\beta_a\rangle` that **must share a common
    displacement vector** :math:`\{\beta_a\}`. The environment is a
    single coherent-state superposition ``env`` :math:`= \sum_j Y_j
    |\sigma_j\rangle`.

    The beamsplitter mixes system and environment with transmissivity
    :math:`\eta`; pure photon loss with rate :math:`\gamma` is the
    special case of vacuum environment (``env = CoherentKet([1], [0])``)
    and :math:`\eta = 1 - \gamma`.

    Parameters
    ----------
    logical_kets : Sequence[CoherentKet]
        Length-:math:`D` encoding. Each ket's ``ds`` must equal
        ``logical_kets[0].ds``.
    env : CoherentKet
        Environment-mode initial state.
    eta : float or Array
        Beamsplitter transmissivity.

    Returns
    -------
    rho_out : Array, shape ``(D, D, A * N_E, A * N_E)``
        Choi-like tensor describing
        :math:`\mathcal{E}(|\mu_L\rangle\!\langle\nu_L|)` in the output
        coherent basis.
    d_out : Array, shape ``(A * N_E,)``
        Output coherent-state displacements,
        :math:`d_{aj} = \sqrt{\eta}\,\beta_a + \sqrt{1-\eta}\,\sigma_j`.

    Raises
    ------
    ValueError
        If the logical kets do not share a common displacement vector
        (checked at Python time; not inside ``jax.jit``).

    Notes
    -----
    For the optimizer hot path, call :func:`_beamsplit_full_arrays`
    directly with raw arrays instead of constructing
    :class:`CoherentKet`\ s every step.
    """
    if len(logical_kets) == 0:
        raise ValueError("logical_kets must contain at least one CoherentKet")
    beta = logical_kets[0].ds
    for k, lk in enumerate(logical_kets[1:], start=1):
        if lk.ds.shape != beta.shape:
            raise ValueError(
                f"logical_kets[{k}].ds has shape {lk.ds.shape} but expected {beta.shape}"
            )
    alpha = jnp.stack([lk.cs for lk in logical_kets])  # (D, A)
    return _beamsplit_full_arrays(alpha, beta, env.cs, env.ds, jnp.asarray(eta))


# ---------------------------------------------------------------------------
# Floating-basis logical encoder
#
# Parametrize an isometric encoder D -> A of a logical D-dim Hilbert space into
# a coherent-state superposition basis by an unconstrained matrix X: (A, D)
# and a displacement vector d: (A,). The algebraic isometry construction
#
#     C = G^{-1/2}|_supp  X  (X^dagger X)^{-1/2}|_supp,   G_{ab} = <d_a|d_b>
#
# guarantees C^dagger G C = I_D (the encoded logical states are orthonormal),
# so the parametrization is unconstrained and well-suited to gradient
# optimization.
# ---------------------------------------------------------------------------


@jax.jit
def unitary_encoding_map(X: Array, d: Array, psi_logical: Array) -> Array:
    r"""Apply the floating-basis isometric encoder to a logical ket.

    For a logical state :math:`|\psi_L\rangle = \sum_\mu \psi_\mu
    |\mu\rangle` in a :math:`D`-dimensional logical space, returns the
    coherent-basis coefficient vector

    .. math::

        c_a = \sum_\mu C_{a\mu}\,\psi_\mu,
        \quad
        C = G^{-1/2}|_{\mathrm{supp}}\,X\,(X^\dagger X)^{-1/2}|_{\mathrm{supp}}

    so that :math:`C^\dagger G C = I_D` exactly. The encoded logical state
    is :math:`\sum_a c_a |d_a\rangle`; wrap with
    :func:`encode_logical_ket` to get a :class:`CoherentKet`.

    Parameters
    ----------
    X : Array, shape ``(A, D)``
        Unconstrained complex parameter matrix.
    d : Array, shape ``(A,)``
        Coherent-basis displacements.
    psi_logical : Array, shape ``(..., D)``
        Logical-basis amplitudes. Leading dims are broadcast.

    Returns
    -------
    Array, shape ``(..., A)``
        Coherent-basis coefficients :math:`c_a` of the encoded state(s).
    """
    A = d.shape[0]
    G = coherent_overlap(d.reshape((A, 1)), d.reshape((1, A)))
    Q = X @ invsqrtm_supp(dag(X) @ X)  # (A, D)
    C = invsqrtm_supp(G) @ Q           # (A, D)
    return jnp.einsum("...l,al->...a", psi_logical, C)


def encode_logical_ket(X: Array, d: Array, mu: int = 0, D: int | None = None) -> CoherentKet:
    r"""Build the encoded logical state :math:`|\psi_\mu\rangle` as a :class:`CoherentKet`.

    Equivalent to ``CoherentKet(cs=unitary_encoding_map(X, d, e_mu), ds=d)``
    where :math:`e_\mu` is the :math:`\mu`-th computational-basis vector
    of the logical :math:`D`-dim space.

    Parameters
    ----------
    X : Array, shape ``(A, D)``
        Encoder parameter matrix.
    d : Array, shape ``(A,)``
        Coherent-basis displacements.
    mu : int
        Logical basis index to encode.
    D : int or None
        Logical dimension. Defaults to ``X.shape[1]``.

    Returns
    -------
    CoherentKet
        Encoded logical state. By construction this is unit-norm, so the
        :class:`CoherentKet` constructor's normalization is a no-op.
    """
    if D is None:
        D = int(X.shape[1])
    psi_mu = jnp.zeros(D, dtype=jnp.complex128).at[mu].set(1.0)
    return CoherentKet(cs=unitary_encoding_map(X, d, psi_mu), ds=d)


def encode_logical_kets(X: Array, d: Array) -> list[CoherentKet]:
    r"""Build all :math:`D` encoded logical states as a list of :class:`CoherentKet`\ s.

    Convenience wrapper around :func:`encode_logical_ket`. All returned
    kets share ``ds = d``, which is the precondition for passing them to
    :func:`beamsplit_full`.

    Parameters
    ----------
    X : Array, shape ``(A, D)``
    d : Array, shape ``(A,)``

    Returns
    -------
    list[CoherentKet]
        Length ``D``; ``result[mu]`` is the encoded :math:`|\mu_L\rangle`.
    """
    D = int(X.shape[1])
    return [encode_logical_ket(X, d, mu=mu, D=D) for mu in range(D)]
