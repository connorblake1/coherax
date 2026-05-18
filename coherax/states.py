"""Quantum state representations.

.. warning::
    This API is under active development and may change without notice.

Provides abstract :class:`Ket` and :class:`DM` base classes with concrete
implementations :class:`CoherentKet`, :class:`CoherentDM`, :class:`FockKet`,
:class:`FockDM`, :class:`QubitKet`, :class:`JointKet`, the :class:`Operator`,
:class:`Displacer`, :class:`Rotator`, :class:`CPTP` operators, and
:class:`BosonicSubspace`.
"""

from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from jaxtyping import Array

from coherax.operators import (
    GKP_N,
    aOmegab,
    coherent_overlap,
    dag,
    dqcoherent,
    sparse_eigh,
)


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
            -0.5 * _abs_sq(db - da - u) + 1j * (aOmegab(da, db) + aOmegab(u, da + db))
        )
        return jnp.sum(jnp.conj(ca) * cb * envelope)

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

        The ordering convention is :math:`|n\rangle\otimes|\mu\rangle` with
        the bosonic index running fastest:

        .. math::

            |\Psi\rangle_{\mathrm{Fock}}
            = \begin{pmatrix}
                \psi_{\mu=0} \\ \psi_{\mu=1}
              \end{pmatrix}
            \in \mathbb{C}^{2N}

        Parameters
        ----------
        N : int
            Fock-space truncation dimension for the bosonic mode.

        Returns
        -------
        Array, shape ``(2*N,)``
        """
        psi = jnp.zeros(2 * N, dtype=jnp.complex128)
        for mu in range(2):
            coherents = jax.vmap(lambda alpha: dqcoherent(N, alpha))(self.ds[mu])
            psi_mu = jnp.einsum("ija,i->ja", coherents, self.cs[mu]).squeeze()
            psi = psi.at[mu * N:(mu + 1) * N].set(psi_mu)
        return psi

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


def _weighted_sum_kets(kets: list[Ket], weights: list[Array]) -> Ket:
    """Compute the weighted superposition :math:`\\sum_i w_i |\\psi_i\\rangle`.

    All kets must be the same concrete type.

    Parameters
    ----------
    kets : list[Ket]
        Basis kets (all the same type).
    weights : list[Array]
        Complex scalar weights.

    Returns
    -------
    Ket
        Combined, normalized ket.
    """
    if all(isinstance(k, CoherentKet) for k in kets):
        combined_ds = jnp.concatenate([k.ds for k in kets])
        combined_cs = jnp.concatenate([w * k.cs for w, k in zip(weights, kets)])
        return CoherentKet(cs=combined_cs, ds=combined_ds)
    if all(isinstance(k, FockKet) for k in kets):
        combined_ns = jnp.concatenate([k.ns for k in kets])
        combined_cs = jnp.concatenate([w * k.cs for w, k in zip(weights, kets)])
        return FockKet(cs=combined_cs, ns=combined_ns)
    raise TypeError(
        "All target kets must be the same concrete type (CoherentKet or FockKet)"
    )


class Operator(eqx.Module):
    r"""Operator :math:`O = \sum_i |\phi_i\rangle\langle\psi_i|`.

    Built from two lists of kets spanning the domain and codomain Hilbert
    spaces.  Applying the operator to a ket :math:`|\xi\rangle` computes

    .. math::
        O|\xi\rangle = \sum_i \langle\psi_i|\xi\rangle\,|\phi_i\rangle

    Parameters
    ----------
    kets_from : list[Ket]
        Orthonormal basis kets for the domain (the :math:`|\psi_i\rangle`).
    kets_to : list[Ket]
        Orthonormal basis kets for the codomain (the :math:`|\phi_i\rangle`).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from coherax.states import FockKet, CoherentKet, Operator
    >>> # Embed |0>,|1> into a coherent-state codespace
    >>> basis_from = [FockKet(cs=jnp.array([1.0]), ns=jnp.array([0])),
    ...              FockKet(cs=jnp.array([1.0]), ns=jnp.array([1]))]
    >>> alpha = 2.0
    >>> basis_to = [CoherentKet(cs=jnp.array([1.0, 1.0]),
    ...                         ds=jnp.array([alpha, -alpha])),
    ...            CoherentKet(cs=jnp.array([1.0, -1.0]),
    ...                         ds=jnp.array([alpha, -alpha]))]
    >>> op = Operator(kets_from=basis_from, kets_to=basis_to)
    >>> result = op.apply(basis_from[0])
    """

    kets_from: list[Ket]
    kets_to: list[Ket]

    def __init__(self, kets_from: list[Ket], kets_to: list[Ket]) -> None:
        if len(kets_from) != len(kets_to):
            raise ValueError(
                f"kets_from ({len(kets_from)}) and kets_to "
                f"({len(kets_to)}) must have the same length"
            )
        self.kets_from = list(kets_from)
        self.kets_to = list(kets_to)

    def apply(self, psi: Ket) -> Ket:
        r"""Apply the operator: :math:`O|\psi\rangle = \sum_i \langle\psi_i|\psi\rangle\,|\phi_i\rangle`.

        Parameters
        ----------
        psi : Ket
            Input ket in the domain Hilbert space.

        Returns
        -------
        Ket
            Output ket in the codomain Hilbert space (normalized).
        """
        weights = [kf.inner(psi) for kf in self.kets_from]
        return _weighted_sum_kets(self.kets_to, weights)

    def apply_adj(self, psi: Ket) -> Ket:
        r"""Apply the adjoint: :math:`O^\dagger|\psi\rangle = \sum_i \langle\phi_i|\psi\rangle\,|\psi_i\rangle`.

        Parameters
        ----------
        psi : Ket
            Input ket in the codomain Hilbert space.

        Returns
        -------
        Ket
            Output ket in the domain Hilbert space (normalized).
        """
        weights = [kt.inner(psi) for kt in self.kets_to]
        return _weighted_sum_kets(self.kets_from, weights)

    def dagger(self) -> Operator:
        r"""Return the adjoint operator :math:`O^\dagger`.

        Returns
        -------
        Operator
        """
        return Operator(kets_from=self.kets_to, kets_to=self.kets_from)

    def apply_dm(self, rho: DM) -> DM:
        r"""Apply :math:`O \rho O^\dagger`.

        .. math::

            O \rho O^\dagger

        Only implemented for Fock-basis conversion; raises
        :class:`NotImplementedError` for basis-defined operators acting
        on coherent-basis density matrices.

        Parameters
        ----------
        rho : DM
            Input density matrix.

        Returns
        -------
        DM

        Raises
        ------
        NotImplementedError
            Always, since basis-defined operators require Fock-space
            conversion for density matrix application.
        """
        raise NotImplementedError(
            "apply_dm on basis Operator requires Fock-space conversion"
        )


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

    def op_c2o_transform(self, Op: Array) -> Array:
        """Transform an operator from the coherent basis to the orthonormal basis.

        Parameters
        ----------
        O : Array, shape ``(A, A)``

        Returns
        -------
        Array, shape ``(K, K)``
        """
        return jnp.einsum("ia,ab,jb->ij", self.Tp, Op, jnp.conj(self.Tp))

    def op_o2c_transform(self, O: Array) -> Array:
        """Transform an operator from the orthonormal basis to the coherent basis.

        Parameters
        ----------
        O : Array, shape ``(K, K)``

        Returns
        -------
        Array, shape ``(A, A)``
        """
        return jnp.einsum("ai,ij,bj->ab", self.T, O, jnp.conj(self.T))

    def ket_c2o_transform(self, ket: Array) -> Array:
        """Transform a ket from the coherent basis to the orthonormal basis.

        Parameters
        ----------
        ket : Array, shape ``(A,)``

        Returns
        -------
        Array, shape ``(K,)``
        """
        return jnp.einsum("ia,a->i", self.Tp, ket)

    def ket_o2c_transform(self, ket: Array) -> Array:
        """Transform a ket from the orthonormal basis to the coherent basis.

        Parameters
        ----------
        ket : Array, shape ``(K,)``

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
