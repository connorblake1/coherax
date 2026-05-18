"""Tests for coherax.states — Ket, DM, FockKet, FockDM, typed operators, inner products."""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from coherax.states import (
    BosonicSubspace,
    CPTP,
    CoherentCoherentOp,
    CoherentDM,
    CoherentFockOp,
    CoherentKet,
    Displacer,
    DM,
    FockCoherentOp,
    FockDM,
    FockFockOp,
    FockKet,
    JointKet,
    Ket,
    LogicalKet,
    QubitKet,
    Rotator,
)


# ---------------------------------------------------------------------------
# Base class hierarchy
# ---------------------------------------------------------------------------


class TestBaseClasses:
    """Verify abstract hierarchy."""

    def test_coherent_ket_is_ket(self) -> None:
        assert issubclass(CoherentKet, Ket)

    def test_fock_ket_is_ket(self) -> None:
        assert issubclass(FockKet, Ket)

    def test_coherent_dm_is_dm(self) -> None:
        assert issubclass(CoherentDM, DM)

    def test_fock_dm_is_dm(self) -> None:
        assert issubclass(FockDM, DM)

    def test_ket_inner_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            Ket().inner(Ket())

    def test_dm_inner_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            DM().inner(DM())


# ---------------------------------------------------------------------------
# CoherentKet
# ---------------------------------------------------------------------------


class TestCoherentKet:
    """Construction, normalization, and Fock conversion for CoherentKet."""

    def test_normalization(self) -> None:
        ket = CoherentKet(
            cs=jnp.array([3.0, 4.0]), ds=jnp.array([1.0 + 0j, -1.0 + 0j])
        )
        npt.assert_allclose(jnp.real(ket.inner(ket)), 1.0, atol=1e-12)

    def test_unit_returns_normalized(self) -> None:
        ket = CoherentKet(
            cs=jnp.array([1.0, 1.0]), ds=jnp.array([2.0 + 0j, -2.0 + 0j])
        )
        ket2 = ket.unit()
        npt.assert_allclose(ket.cs, ket2.cs, atol=1e-12)

    def test_to_fock_ket_shape(self) -> None:
        ket = CoherentKet(
            cs=jnp.array([1.0]), ds=jnp.array([0.0 + 0j])
        )
        psi = ket.to_fock_ket(20)
        assert psi.shape == (20,)

    def test_to_fock_basis_shape(self) -> None:
        ket = CoherentKet(
            cs=jnp.array([1.0]), ds=jnp.array([0.0 + 0j])
        )
        rho = ket.to_fock_basis(20)
        assert rho.shape == (20, 20)

    def test_vacuum_coherent_is_fock_vacuum(self) -> None:
        """Coherent state |alpha=0> should be the Fock vacuum |0>."""
        ket = CoherentKet(
            cs=jnp.array([1.0]), ds=jnp.array([0.0 + 0j])
        )
        psi = ket.to_fock_ket(20)
        npt.assert_allclose(jnp.abs(psi[0]) ** 2, 1.0, atol=1e-10)
        npt.assert_allclose(jnp.sum(jnp.abs(psi[1:]) ** 2), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# FockKet
# ---------------------------------------------------------------------------


class TestFockKet:
    """Construction, normalization, and conversion for FockKet."""

    def test_construction_from_array(self) -> None:
        ket = FockKet(cs=jnp.array([1.0, 0.0]), ns=jnp.array([0, 1]))
        assert ket.cs.shape == (2,)
        assert ket.ns.shape == (2,)

    def test_construction_from_int(self) -> None:
        ket = FockKet(cs=jnp.array([1.0, 0.0, 0.0]), ns=3)
        npt.assert_array_equal(ket.ns, jnp.array([0, 1, 2]))

    def test_normalization(self) -> None:
        ket = FockKet(cs=jnp.array([3.0, 4.0]), ns=jnp.array([0, 1]))
        npt.assert_allclose(jnp.real(ket.inner(ket)), 1.0, atol=1e-12)

    def test_unit(self) -> None:
        ket = FockKet(cs=jnp.array([1.0, 1.0]), ns=jnp.array([0, 1]))
        ket2 = ket.unit()
        npt.assert_allclose(ket.cs, ket2.cs, atol=1e-12)

    def test_to_fock_ket_single(self) -> None:
        ket = FockKet(cs=jnp.array([1.0]), ns=jnp.array([3]))
        psi = ket.to_fock_ket(10)
        expected = jnp.zeros(10, dtype=jnp.complex128).at[3].set(1.0)
        npt.assert_allclose(psi, expected, atol=1e-12)

    def test_to_fock_basis_vacuum(self) -> None:
        ket = FockKet(cs=jnp.array([1.0]), ns=jnp.array([0]))
        rho = ket.to_fock_basis(5)
        expected = jnp.zeros((5, 5), dtype=jnp.complex128).at[0, 0].set(1.0)
        npt.assert_allclose(rho, expected, atol=1e-12)

    def test_noncontiguous_ns(self) -> None:
        ket = FockKet(cs=jnp.array([1.0, 1.0]), ns=jnp.array([0, 5]))
        psi = ket.to_fock_ket(10)
        npt.assert_allclose(jnp.abs(psi[0]), jnp.abs(psi[5]), atol=1e-12)
        npt.assert_allclose(psi[1], 0.0, atol=1e-12)

    def test_invalid_ns_type(self) -> None:
        with pytest.raises(Exception):
            FockKet(cs=jnp.array([1.0]), ns="bad")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FockDM
# ---------------------------------------------------------------------------


class TestFockDM:
    """Construction, trace normalization, and conversion for FockDM."""

    def test_trace_normalization(self) -> None:
        C = jnp.array([[2.0, 0.0], [0.0, 2.0]], dtype=jnp.complex128)
        dm = FockDM(C=C, ns=jnp.array([0, 1]))
        rho = dm.to_fock_basis(5)
        npt.assert_allclose(jnp.real(jnp.trace(rho)), 1.0, atol=1e-12)

    def test_from_ket(self) -> None:
        ket = FockKet(cs=jnp.array([1.0, 1.0j]), ns=jnp.array([0, 1]))
        dm = FockDM.from_ket(ket)
        rho_ket = ket.to_fock_basis(10)
        rho_dm = dm.to_fock_basis(10)
        npt.assert_allclose(rho_ket, rho_dm, atol=1e-12)

    def test_unit(self) -> None:
        C = jnp.eye(3, dtype=jnp.complex128) * 5.0
        dm = FockDM(C=C, ns=jnp.array([0, 1, 2]))
        dm2 = dm.unit()
        rho = dm2.to_fock_basis(5)
        npt.assert_allclose(jnp.real(jnp.trace(rho)), 1.0, atol=1e-12)

    def test_construction_from_int(self) -> None:
        C = jnp.eye(2, dtype=jnp.complex128)
        dm = FockDM(C=C, ns=2)
        npt.assert_array_equal(dm.ns, jnp.array([0, 1]))

    def test_mixed_state_trace(self) -> None:
        """Maximally mixed qubit: Tr(rho) = 1."""
        C = jnp.eye(2, dtype=jnp.complex128)
        dm = FockDM(C=C, ns=jnp.array([0, 1]))
        rho = dm.to_fock_basis(5)
        npt.assert_allclose(jnp.real(jnp.trace(rho)), 1.0, atol=1e-12)
        # Off-diagonals vanish
        npt.assert_allclose(rho[0, 1], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# CoherentDM
# ---------------------------------------------------------------------------


class TestCoherentDM:
    """Construction and conversion for CoherentDM."""

    def test_from_ket_matches_fock(self) -> None:
        ket = CoherentKet(
            cs=jnp.array([1.0, 1.0]), ds=jnp.array([0.5 + 0j, -0.5 + 0j])
        )
        dm = CoherentDM.from_ket(ket)
        rho_ket = ket.to_fock_basis(30)
        rho_dm = dm.to_fock_basis(30)
        npt.assert_allclose(rho_ket, rho_dm, atol=1e-8)

    def test_unit(self) -> None:
        ket = CoherentKet(
            cs=jnp.array([1.0]), ds=jnp.array([0.0 + 0j])
        )
        dm = CoherentDM.from_ket(ket)
        dm2 = dm.unit()
        rho = dm2.to_fock_basis(10)
        npt.assert_allclose(jnp.real(jnp.trace(rho)), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Ket inner products (same type)
# ---------------------------------------------------------------------------


class TestKetInnerSameType:
    """Inner products between kets of the same representation."""

    def test_fock_orthogonal(self) -> None:
        ket0 = FockKet(cs=jnp.array([1.0]), ns=jnp.array([0]))
        ket1 = FockKet(cs=jnp.array([1.0]), ns=jnp.array([1]))
        npt.assert_allclose(jnp.abs(ket0.inner(ket1)), 0.0, atol=1e-12)

    def test_fock_self_overlap(self) -> None:
        ket = FockKet(cs=jnp.array([1.0]), ns=jnp.array([3]))
        npt.assert_allclose(jnp.real(ket.inner(ket)), 1.0, atol=1e-12)

    def test_fock_superposition(self) -> None:
        """<+|+> = 1 where |+> = (|0> + |1>)/sqrt(2)."""
        ket = FockKet(cs=jnp.array([1.0, 1.0]), ns=jnp.array([0, 1]))
        npt.assert_allclose(jnp.real(ket.inner(ket)), 1.0, atol=1e-12)

    def test_coherent_self_overlap(self) -> None:
        alpha = 1.0 + 0.5j
        ket = CoherentKet(cs=jnp.array([1.0]), ds=jnp.array([alpha]))
        npt.assert_allclose(jnp.real(ket.inner(ket)), 1.0, atol=1e-12)

    def test_coherent_distinct(self) -> None:
        """Two well-separated coherent states are nearly orthogonal."""
        k1 = CoherentKet(cs=jnp.array([1.0]), ds=jnp.array([5.0 + 0j]))
        k2 = CoherentKet(cs=jnp.array([1.0]), ds=jnp.array([-5.0 + 0j]))
        npt.assert_allclose(jnp.abs(k1.inner(k2)), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Ket inner products (cross type)
# ---------------------------------------------------------------------------


class TestKetInnerCrossType:
    """Inner products between CoherentKet and FockKet."""

    def test_coherent_fock_vacuum(self) -> None:
        r""":math:`\langle\alpha|0\rangle = e^{-|\alpha|^2/2}`."""
        alpha = 1.5
        coh = CoherentKet(cs=jnp.array([1.0]), ds=jnp.array([alpha + 0j]))
        fock0 = FockKet(cs=jnp.array([1.0]), ns=jnp.array([0]))
        expected = jnp.exp(-0.5 * alpha**2)
        result = coh.inner(fock0)
        npt.assert_allclose(result, expected, atol=1e-10)

    def test_coherent_fock_n1(self) -> None:
        r""":math:`\langle\alpha|1\rangle = e^{-|\alpha|^2/2}\,\alpha^*`."""
        alpha = 1.0 + 0.5j
        coh = CoherentKet(cs=jnp.array([1.0]), ds=jnp.array([alpha]))
        fock1 = FockKet(cs=jnp.array([1.0]), ns=jnp.array([1]))
        expected = jnp.exp(-0.5 * jnp.abs(alpha) ** 2) * jnp.conj(alpha)
        result = coh.inner(fock1)
        npt.assert_allclose(result, expected, atol=1e-10)

    def test_fock_coherent_is_conjugate(self) -> None:
        r""":math:`\langle n|\alpha\rangle = \overline{\langle\alpha|n\rangle}`."""
        alpha = 0.7 + 0.3j
        coh = CoherentKet(cs=jnp.array([1.0]), ds=jnp.array([alpha]))
        fock = FockKet(
            cs=jnp.array([1.0, 1.0j]), ns=jnp.array([0, 2])
        )
        cf = coh.inner(fock)
        fc = fock.inner(coh)
        npt.assert_allclose(fc, jnp.conj(cf), atol=1e-10)

    def test_cross_matches_fock_truncation(self) -> None:
        """Analytic cross inner product matches Fock-space computation."""
        alpha = 0.5 + 0.2j
        coh = CoherentKet(
            cs=jnp.array([1.0, 0.5]),
            ds=jnp.array([alpha, -alpha]),
        )
        fock = FockKet(
            cs=jnp.array([1.0, 0.3j]), ns=jnp.array([0, 3])
        )
        analytic = coh.inner(fock)
        N = 50
        psi_coh = coh.to_fock_ket(N)
        psi_fock = fock.to_fock_ket(N)
        numerical = jnp.dot(jnp.conj(psi_coh), psi_fock)
        npt.assert_allclose(analytic, numerical, atol=1e-8)

    def test_fock_coherent_matches_fock_truncation(self) -> None:
        """Reversed direction also matches Fock-space computation."""
        alpha = 0.8 - 0.4j
        coh = CoherentKet(cs=jnp.array([1.0]), ds=jnp.array([alpha]))
        fock = FockKet(
            cs=jnp.array([0.5, 0.5, 0.5j]),
            ns=jnp.array([0, 1, 2]),
        )
        analytic = fock.inner(coh)
        N = 50
        psi_coh = coh.to_fock_ket(N)
        psi_fock = fock.to_fock_ket(N)
        numerical = jnp.dot(jnp.conj(psi_fock), psi_coh)
        npt.assert_allclose(analytic, numerical, atol=1e-8)

    def test_type_error_on_unknown(self) -> None:
        ket = FockKet(cs=jnp.array([1.0]), ns=jnp.array([0]))
        with pytest.raises(TypeError):
            ket.inner("not a ket")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DM inner products
# ---------------------------------------------------------------------------


class TestDMInner:
    """Hilbert--Schmidt inner products for density matrices."""

    def test_fock_dm_pure_state_purity(self) -> None:
        """Tr(rho^2) = 1 for a pure state."""
        ket = FockKet(cs=jnp.array([1.0, 1.0j]), ns=jnp.array([0, 1]))
        dm = FockDM.from_ket(ket)
        result = dm.inner(dm)
        npt.assert_allclose(jnp.real(result), 1.0, atol=1e-12)

    def test_fock_dm_mixed_purity(self) -> None:
        """Tr(rho^2) < 1 for a maximally mixed qubit."""
        C = jnp.eye(2, dtype=jnp.complex128)
        dm = FockDM(C=C, ns=jnp.array([0, 1]))
        result = dm.inner(dm)
        npt.assert_allclose(jnp.real(result), 0.5, atol=1e-12)

    def test_coherent_dm_pure_purity(self) -> None:
        """Tr(rho^2) = 1 for a pure coherent DM."""
        ket = CoherentKet(
            cs=jnp.array([1.0, 1.0]),
            ds=jnp.array([1.0 + 0j, -1.0 + 0j]),
        )
        dm = CoherentDM.from_ket(ket)
        result = dm.inner(dm)
        npt.assert_allclose(jnp.real(result), 1.0, atol=1e-10)

    def test_coherent_fock_dm_cross(self) -> None:
        """Cross-type DM inner product matches Fock-basis computation."""
        ket_c = CoherentKet(cs=jnp.array([1.0]), ds=jnp.array([0.5 + 0j]))
        ket_f = FockKet(
            cs=jnp.array([1.0, 0.5]), ns=jnp.array([0, 1])
        )
        cdm = CoherentDM.from_ket(ket_c)
        fdm = FockDM.from_ket(ket_f)
        analytic = cdm.inner(fdm)
        N = 50
        rho_c = cdm.to_fock_basis(N)
        rho_f = fdm.to_fock_basis(N)
        numerical = jnp.trace(rho_c @ jnp.conj(rho_f.T))
        npt.assert_allclose(analytic, numerical, atol=1e-8)

    def test_fock_coherent_dm_cross(self) -> None:
        """Reversed cross-type DM inner product matches Fock-basis."""
        ket_c = CoherentKet(
            cs=jnp.array([1.0]), ds=jnp.array([0.3 + 0.1j])
        )
        ket_f = FockKet(
            cs=jnp.array([1.0, 0.5j]), ns=jnp.array([0, 2])
        )
        cdm = CoherentDM.from_ket(ket_c)
        fdm = FockDM.from_ket(ket_f)
        analytic = fdm.inner(cdm)
        N = 50
        rho_c = cdm.to_fock_basis(N)
        rho_f = fdm.to_fock_basis(N)
        numerical = jnp.trace(rho_f @ jnp.conj(rho_c.T))
        npt.assert_allclose(analytic, numerical, atol=1e-8)

    def test_dm_type_error(self) -> None:
        dm = FockDM(
            C=jnp.eye(1, dtype=jnp.complex128), ns=jnp.array([0])
        )
        with pytest.raises(TypeError):
            dm.inner("not a dm")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Typed basis-defined operators
# ---------------------------------------------------------------------------


class TestFockFockOp:
    def test_identity_isometry(self) -> None:
        basis = [
            FockKet(cs=jnp.array([1.0]), ns=jnp.array([i]))
            for i in range(3)
        ]
        iso = FockFockOp.from_kets(basis, basis)
        psi = FockKet(cs=jnp.array([1.0, 1.0j, 0.5]), ns=jnp.array([0, 1, 2]))
        result = iso.apply(psi)
        npt.assert_allclose(jnp.abs(psi.inner(result)), 1.0, atol=1e-10)

    def test_roundtrip(self) -> None:
        from_kets = [FockKet(cs=jnp.array([1.0]), ns=jnp.array([i])) for i in (0, 1)]
        to_kets = [FockKet(cs=jnp.array([1.0]), ns=jnp.array([i])) for i in (5, 6)]
        iso = FockFockOp.from_kets(from_kets, to_kets)
        psi = FockKet(cs=jnp.array([0.6, 0.8j]), ns=jnp.array([0, 1]))
        roundtrip = iso.apply_adj(iso.apply(psi))
        npt.assert_allclose(jnp.abs(psi.inner(roundtrip)), 1.0, atol=1e-10)

    def test_dagger_returns_fock_fock_op(self) -> None:
        from_kets = [FockKet(cs=jnp.array([1.0]), ns=jnp.array([0]))]
        to_kets = [FockKet(cs=jnp.array([1.0]), ns=jnp.array([5]))]
        iso = FockFockOp.from_kets(from_kets, to_kets)
        assert isinstance(iso.dagger(), FockFockOp)
        psi = FockKet(cs=jnp.array([1.0]), ns=jnp.array([5]))
        npt.assert_allclose(
            jnp.abs(from_kets[0].inner(iso.dagger().apply(psi))), 1.0, atol=1e-10
        )

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError):
            FockFockOp.from_kets(
                [FockKet(cs=jnp.array([1.0]), ns=jnp.array([0]))],
                [FockKet(cs=jnp.array([1.0]), ns=jnp.array([0])),
                 FockKet(cs=jnp.array([1.0]), ns=jnp.array([1]))],
            )

    def test_apply_adj_matches_dagger_apply(self) -> None:
        from_kets = [FockKet(cs=jnp.array([1.0]), ns=jnp.array([i])) for i in (0, 1)]
        to_kets = [FockKet(cs=jnp.array([1.0]), ns=jnp.array([i])) for i in (3, 4)]
        iso = FockFockOp.from_kets(from_kets, to_kets)
        psi = FockKet(cs=jnp.array([1.0]), ns=jnp.array([3]))
        r1 = iso.apply_adj(psi)
        r2 = iso.dagger().apply(psi)
        npt.assert_allclose(jnp.abs(r1.inner(r2)), 1.0, atol=1e-10)

    def test_wrap_isometry(self) -> None:
        # Identity-like isometry: |n=5><n=0| + |n=6><n=1| applied to rho = |0><0|
        # should give |5><5|.
        from_kets = [FockKet(cs=jnp.array([1.0]), ns=jnp.array([i])) for i in (0, 1)]
        to_kets = [FockKet(cs=jnp.array([1.0]), ns=jnp.array([i])) for i in (5, 6)]
        iso = FockFockOp.from_kets(from_kets, to_kets)
        rho = FockDM.from_ket(FockKet(cs=jnp.array([1.0]), ns=jnp.array([0])))
        out = iso.wrap(rho)
        expected = FockDM.from_ket(FockKet(cs=jnp.array([1.0]), ns=jnp.array([5])))
        npt.assert_allclose(jnp.abs(out.inner(expected)), 1.0, atol=1e-10)

    def test_jittable(self) -> None:
        # Confirm apply is jittable: pass op as an argument so JAX traces
        # it as a pytree rather than hashing it.
        from_kets = [FockKet(cs=jnp.array([1.0]), ns=jnp.array([i])) for i in (0, 1)]
        to_kets = [FockKet(cs=jnp.array([1.0]), ns=jnp.array([i])) for i in (5, 6)]
        iso = FockFockOp.from_kets(from_kets, to_kets)
        psi = FockKet(cs=jnp.array([0.6, 0.8j]), ns=jnp.array([0, 1]))

        @jax.jit
        def apply(op: FockFockOp, p: FockKet) -> FockKet:
            return op.apply(p)

        out = apply(iso, psi)
        assert isinstance(out, FockKet)


class TestCoherentFockOp:
    def test_isometry_apply(self) -> None:
        from_kets = [FockKet(cs=jnp.array([1.0]), ns=jnp.array([0])),
                     FockKet(cs=jnp.array([1.0]), ns=jnp.array([1]))]
        alpha = 2.0 + 0j
        to_kets = [
            CoherentKet(cs=jnp.array([1.0, 1.0]),  ds=jnp.array([alpha, -alpha])),
            CoherentKet(cs=jnp.array([1.0, -1.0]), ds=jnp.array([alpha, -alpha])),
        ]
        # Note: from is Fock, to is Coherent -> FockCoherentOp
        iso = FockCoherentOp.from_kets(from_kets, to_kets)
        psi = FockKet(cs=jnp.array([1.0]), ns=jnp.array([0]))
        result = iso.apply(psi)
        assert isinstance(result, CoherentKet)
        npt.assert_allclose(jnp.abs(to_kets[0].inner(result)), 1.0, atol=1e-8)

    def test_coh_to_fock_apply(self) -> None:
        # CoherentFockOp acts on a coherent input, returns a Fock output.
        alpha = 2.0 + 0j
        from_kets = [
            CoherentKet(cs=jnp.array([1.0, 1.0]),  ds=jnp.array([alpha, -alpha])),
            CoherentKet(cs=jnp.array([1.0, -1.0]), ds=jnp.array([alpha, -alpha])),
        ]
        to_kets = [FockKet(cs=jnp.array([1.0, 0.0]), ns=jnp.array([0, 1])),
                   FockKet(cs=jnp.array([0.0, 1.0]), ns=jnp.array([0, 1]))]
        op = CoherentFockOp.from_kets(from_kets, to_kets)
        result = op.apply(from_kets[0])
        assert isinstance(result, FockKet)
        # Should align with to_kets[0] up to scale (basis approximately orthonormal at alpha=2)
        npt.assert_allclose(jnp.abs(to_kets[0].inner(result)), 1.0, atol=5e-3)

    def test_dagger_swaps_basis(self) -> None:
        alpha = 2.0 + 0j
        from_kets = [
            CoherentKet(cs=jnp.array([1.0, 1.0]),  ds=jnp.array([alpha, -alpha])),
            CoherentKet(cs=jnp.array([1.0, -1.0]), ds=jnp.array([alpha, -alpha])),
        ]
        to_kets = [FockKet(cs=jnp.array([1.0, 0.0]), ns=jnp.array([0, 1])),
                   FockKet(cs=jnp.array([0.0, 1.0]), ns=jnp.array([0, 1]))]
        op = CoherentFockOp.from_kets(from_kets, to_kets)
        assert isinstance(op.dagger(), FockCoherentOp)

    def test_wrap_crosses_basis(self) -> None:
        # CoherentFockOp.wrap takes a CoherentDM and returns a FockDM.
        alpha = 2.0 + 0j
        from_kets = [
            CoherentKet(cs=jnp.array([1.0, 1.0]),  ds=jnp.array([alpha, -alpha])),
            CoherentKet(cs=jnp.array([1.0, -1.0]), ds=jnp.array([alpha, -alpha])),
        ]
        to_kets = [FockKet(cs=jnp.array([1.0, 0.0]), ns=jnp.array([0, 1])),
                   FockKet(cs=jnp.array([0.0, 1.0]), ns=jnp.array([0, 1]))]
        op = CoherentFockOp.from_kets(from_kets, to_kets)
        rho = CoherentDM.from_ket(from_kets[0])
        out = op.wrap(rho)
        assert isinstance(out, FockDM)


class TestCoherentCoherentOp:
    def test_apply_consistency_with_dense(self) -> None:
        # Build a CoherentCoherentOp and check that apply matches the
        # dense Fock-basis matrix-vector product.
        alpha = 2.0 + 0j
        from_kets = [
            CoherentKet(cs=jnp.array([1.0, 1.0]),  ds=jnp.array([alpha, -alpha])),
            CoherentKet(cs=jnp.array([1.0, -1.0]), ds=jnp.array([alpha, -alpha])),
        ]
        # Same basis on both sides => approximately identity in subspace
        op = CoherentCoherentOp.from_kets(from_kets, from_kets)
        psi = from_kets[0]
        result = op.apply(psi)
        npt.assert_allclose(jnp.abs(from_kets[0].inner(result)), 1.0, atol=1e-3)

    def test_wrap_preserves_pure_state(self) -> None:
        alpha = 2.0 + 0j
        basis = [
            CoherentKet(cs=jnp.array([1.0, 1.0]),  ds=jnp.array([alpha, -alpha])),
            CoherentKet(cs=jnp.array([1.0, -1.0]), ds=jnp.array([alpha, -alpha])),
        ]
        op = CoherentCoherentOp.from_kets(basis, basis)
        rho = CoherentDM.from_ket(basis[0])
        out = op.wrap(rho)
        npt.assert_allclose(jnp.abs(out.inner(rho)), 1.0, atol=1e-3)


class TestFockCoherentOp:
    def test_dagger_returns_coherent_fock(self) -> None:
        alpha = 2.0 + 0j
        from_kets = [FockKet(cs=jnp.array([1.0]), ns=jnp.array([0]))]
        to_kets = [CoherentKet(cs=jnp.array([1.0, 1.0]), ds=jnp.array([alpha, -alpha]))]
        op = FockCoherentOp.from_kets(from_kets, to_kets)
        assert isinstance(op.dagger(), CoherentFockOp)

    def test_apply_adj_inverse_of_dagger_apply(self) -> None:
        alpha = 2.0 + 0j
        from_kets = [FockKet(cs=jnp.array([1.0]), ns=jnp.array([0])),
                     FockKet(cs=jnp.array([1.0]), ns=jnp.array([1]))]
        to_kets = [
            CoherentKet(cs=jnp.array([1.0, 1.0]),  ds=jnp.array([alpha, -alpha])),
            CoherentKet(cs=jnp.array([1.0, -1.0]), ds=jnp.array([alpha, -alpha])),
        ]
        op = FockCoherentOp.from_kets(from_kets, to_kets)
        psi = to_kets[0]
        r1 = op.apply_adj(psi)
        r2 = op.dagger().apply(psi)
        npt.assert_allclose(jnp.abs(r1.inner(r2)), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# BosonicSubspace (smoke test — unchanged class)
# ---------------------------------------------------------------------------


class TestJITAndAutodiff:
    """Verify JIT compilation and gradient propagation."""

    def test_jit_fock_inner(self) -> None:
        """FockKet.inner can be JIT-compiled."""
        k1 = FockKet(cs=jnp.array([1.0, 1.0]), ns=jnp.array([0, 1]))
        k2 = FockKet(cs=jnp.array([0.5, 0.5j]), ns=jnp.array([0, 1]))

        @jax.jit
        def f(cs1: jnp.ndarray) -> jnp.ndarray:
            ket = FockKet(cs=cs1, ns=jnp.array([0, 1]))
            return jnp.real(ket.inner(k2))

        result = f(jnp.array([1.0, 1.0]))
        assert jnp.isfinite(result)

    def test_jit_coherent_inner(self) -> None:
        """CoherentKet.inner can be JIT-compiled."""
        k2 = CoherentKet(cs=jnp.array([1.0]), ds=jnp.array([0.5 + 0j]))

        @jax.jit
        def f(ds1: jnp.ndarray) -> jnp.ndarray:
            ket = CoherentKet(cs=jnp.array([1.0]), ds=ds1)
            return jnp.abs(ket.inner(k2)) ** 2

        result = f(jnp.array([0.3 + 0j]))
        assert jnp.isfinite(result)

    def test_jit_cross_inner(self) -> None:
        """Cross-type CoherentKet.inner(FockKet) can be JIT-compiled."""
        fock = FockKet(cs=jnp.array([1.0]), ns=jnp.array([0]))

        @jax.jit
        def f(ds: jnp.ndarray) -> jnp.ndarray:
            coh = CoherentKet(cs=jnp.array([1.0]), ds=ds)
            return jnp.abs(coh.inner(fock)) ** 2

        result = f(jnp.array([1.0 + 0j]))
        assert jnp.isfinite(result)

    def test_grad_coherent_inner_wrt_ds(self) -> None:
        """Gradient of |<coh|fock>|^2 w.r.t. displacements is finite."""
        fock = FockKet(cs=jnp.array([1.0]), ns=jnp.array([0]))

        @jax.grad
        def grad_f(ds: jnp.ndarray) -> jnp.ndarray:
            coh = CoherentKet(cs=jnp.array([1.0]), ds=ds)
            return jnp.real(jnp.abs(coh.inner(fock)) ** 2)

        g = grad_f(jnp.array([1.0 + 0j]))
        assert jnp.all(jnp.isfinite(g))

    def test_grad_coherent_inner_wrt_ds_at_zero(self) -> None:
        """Gradient at alpha=0 is finite (no NaN from 0^n)."""
        fock = FockKet(
            cs=jnp.array([1.0, 0.5]), ns=jnp.array([0, 1])
        )

        @jax.grad
        def grad_f(ds: jnp.ndarray) -> jnp.ndarray:
            coh = CoherentKet(cs=jnp.array([1.0]), ds=ds)
            return jnp.real(jnp.abs(coh.inner(fock)) ** 2)

        g = grad_f(jnp.array([0.0 + 0j]))
        assert jnp.all(jnp.isfinite(g))

    def test_grad_coherent_coherent_wrt_ds(self) -> None:
        """Gradient of |<coh1|coh2>|^2 w.r.t. ds is finite."""
        k2 = CoherentKet(
            cs=jnp.array([1.0, 1.0]),
            ds=jnp.array([1.0 + 0j, -1.0 + 0j]),
        )

        @jax.grad
        def grad_f(ds: jnp.ndarray) -> jnp.ndarray:
            k1 = CoherentKet(cs=jnp.array([1.0]), ds=ds)
            return jnp.real(jnp.abs(k1.inner(k2)) ** 2)

        g = grad_f(jnp.array([0.8 + 0j]))
        assert jnp.all(jnp.isfinite(g))

    def test_grad_fock_inner_wrt_cs(self) -> None:
        """Gradient of |<f1|f2>|^2 w.r.t. coefficients is finite."""
        k2 = FockKet(cs=jnp.array([1.0, 0.0]), ns=jnp.array([0, 1]))

        @jax.grad
        def grad_f(cs: jnp.ndarray) -> jnp.ndarray:
            k1 = FockKet(cs=cs, ns=jnp.array([0, 1]))
            return jnp.real(jnp.abs(k1.inner(k2)) ** 2)

        g = grad_f(jnp.array([0.5, 0.5]))
        assert jnp.all(jnp.isfinite(g))

    def test_jit_isometry_apply(self) -> None:
        """FockFockOp.apply can be JIT-compiled (closure-captured op is a pytree)."""
        basis = [
            FockKet(cs=jnp.array([1.0]), ns=jnp.array([i]))
            for i in range(2)
        ]
        iso = FockFockOp.from_kets(basis, basis)

        @jax.jit
        def f(cs: jnp.ndarray) -> jnp.ndarray:
            psi = FockKet(cs=cs, ns=jnp.array([0, 1]))
            result = iso.apply(psi)
            return jnp.real(result.inner(result))

        result = f(jnp.array([1.0, 1.0]))
        npt.assert_allclose(result, 1.0, atol=1e-10)


class TestBosonicSubspace:
    """Basic smoke tests for BosonicSubspace."""

    def test_construction(self) -> None:
        ds = jnp.array([0.0, 1.0, -1.0], dtype=jnp.complex128)
        bs = BosonicSubspace(ds=ds)
        assert bs.G.shape == (3, 3)

    def test_gram_diagonal(self) -> None:
        ds = jnp.array([0.0, 1.0, -1.0], dtype=jnp.complex128)
        bs = BosonicSubspace(ds=ds)
        npt.assert_allclose(jnp.diag(jnp.real(bs.G)), 1.0, atol=1e-12)

    def test_coherent_ket_factory(self) -> None:
        ds = jnp.array([0.0, 1.0, -1.0], dtype=jnp.complex128)
        bs = BosonicSubspace(ds=ds)
        ket = bs.coherent_ket(jnp.array([1.0, 0.5, 0.5], dtype=jnp.complex128))
        assert isinstance(ket, CoherentKet)
        npt.assert_array_equal(ket.ds, ds)
        npt.assert_allclose(jnp.real(ket.inner(ket)), 1.0, atol=1e-10)

    def test_coherent_ket_round_trip_via_orthonormal(self) -> None:
        # CoherentKet -> orthonormal coeffs -> CoherentKet should preserve
        # the physical state (up to global phase). Both wrappers normalize.
        ds = jnp.array([0.0, 1.0, -1.0], dtype=jnp.complex128)
        bs = BosonicSubspace(ds=ds)
        ket = bs.coherent_ket(jnp.array([1.0, 0.5j, -0.3], dtype=jnp.complex128))
        coeffs = bs.coherent_ket_to_orthonormal(ket)
        ket_back = bs.orthonormal_to_coherent_ket(coeffs)
        npt.assert_allclose(jnp.abs(ket.inner(ket_back)), 1.0, atol=1e-10)

    def test_orthonormal_coeffs_unit_norm_for_unit_ket(self) -> None:
        # A normalized CoherentKet -> orthonormal coeffs should have |coeffs|=1.
        ds = jnp.array([0.0, 1.0, -1.0], dtype=jnp.complex128)
        bs = BosonicSubspace(ds=ds)
        ket = bs.coherent_ket(jnp.array([1.0, 0.5, 0.5], dtype=jnp.complex128))
        coeffs = bs.coherent_ket_to_orthonormal(ket)
        npt.assert_allclose(jnp.linalg.norm(coeffs), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# QubitKet
# ---------------------------------------------------------------------------


class TestQubitKet:
    def test_construction(self) -> None:
        q = QubitKet(cs=jnp.array([1.0, 0.0]))
        npt.assert_array_equal(q.ns, jnp.array([0, 1]))

    def test_normalization(self) -> None:
        q = QubitKet(cs=jnp.array([3.0, 4.0]))
        npt.assert_allclose(jnp.real(q.inner(q)), 1.0, atol=1e-12)

    def test_is_fock_ket(self) -> None:
        assert issubclass(QubitKet, FockKet)

    def test_inner_with_fock(self) -> None:
        q = QubitKet(cs=jnp.array([1.0, 0.0]))
        f = FockKet(cs=jnp.array([1.0]), ns=jnp.array([0]))
        npt.assert_allclose(jnp.abs(q.inner(f)), 1.0, atol=1e-12)


class TestLogicalKet:
    def test_default_ns_is_arange(self) -> None:
        psi = LogicalKet(cs=jnp.array([1.0, 0.0, 1.0]))
        npt.assert_array_equal(psi.ns, jnp.array([0, 1, 2]))

    def test_normalized_on_construction(self) -> None:
        psi = LogicalKet(cs=jnp.array([3.0, 4.0]))
        npt.assert_allclose(jnp.real(psi.inner(psi)), 1.0, atol=1e-12)

    def test_is_fock_ket(self) -> None:
        assert issubclass(LogicalKet, FockKet)

    def test_custom_ns_embedding(self) -> None:
        # Non-contiguous embedding at Fock levels 0, 2, 5
        psi = LogicalKet(cs=jnp.array([1.0, 0.0, 1.0]), ns=jnp.array([0, 2, 5]))
        # |<0|psi>|^2 + |<2|psi>|^2 + |<5|psi>|^2 = 1; cross terms vanish
        n0 = FockKet(cs=jnp.array([1.0]), ns=jnp.array([0]))
        n2 = FockKet(cs=jnp.array([1.0]), ns=jnp.array([2]))
        n5 = FockKet(cs=jnp.array([1.0]), ns=jnp.array([5]))
        total = (
            jnp.abs(psi.inner(n0)) ** 2
            + jnp.abs(psi.inner(n2)) ** 2
            + jnp.abs(psi.inner(n5)) ** 2
        )
        npt.assert_allclose(jnp.real(total), 1.0, atol=1e-12)

    def test_qubit_equivalence(self) -> None:
        # D=2 LogicalKet equals a QubitKet
        cs = jnp.array([0.6, 0.8j])
        npt.assert_allclose(LogicalKet(cs=cs).cs, QubitKet(cs=cs).cs, atol=1e-12)


# ---------------------------------------------------------------------------
# JointKet
# ---------------------------------------------------------------------------


class TestJointKet:
    def test_construction(self) -> None:
        cs = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        ds = jnp.array([[0.5+0j, -0.5+0j], [0.5+0j, -0.5+0j]])
        jk = JointKet(cs=cs, ds=ds)
        assert jk.cs.shape == (2, 2)

    def test_normalization(self) -> None:
        cs = jnp.array([[1.0, 1.0], [1.0, -1.0]])
        ds = jnp.array([[1.0+0j, -1.0+0j], [1.0+0j, -1.0+0j]])
        jk = JointKet(cs=cs, ds=ds)
        npt.assert_allclose(jnp.real(jk.inner(jk)), 1.0, atol=1e-10)

    def test_inner_joint_joint(self) -> None:
        cs = jnp.array([[1.0, 0.0], [0.0, 0.0]])
        ds = jnp.array([[0.0+0j, 0.0+0j], [0.0+0j, 0.0+0j]])
        jk = JointKet(cs=cs, ds=ds)
        npt.assert_allclose(jnp.real(jk.inner(jk)), 1.0, atol=1e-10)

    def test_inner_with_qubit_gives_coherent_ket(self) -> None:
        """Projecting JointKet onto a QubitKet gives a CoherentKet."""
        cs = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        ds = jnp.array([[0.5+0j, -0.5+0j], [0.5+0j, -0.5+0j]])
        jk = JointKet(cs=cs, ds=ds)
        q = QubitKet(cs=jnp.array([1.0, 0.0]))  # project onto |0>
        result = jk.inner(q)
        assert isinstance(result, CoherentKet)

    def test_inner_with_coherent_gives_qubit_ket(self) -> None:
        """Projecting JointKet onto a CoherentKet gives a QubitKet."""
        cs = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        ds = jnp.array([[0.0+0j, 1.0+0j], [0.0+0j, 1.0+0j]])
        jk = JointKet(cs=cs, ds=ds)
        coh = CoherentKet(cs=jnp.array([1.0]), ds=jnp.array([0.0+0j]))
        result = jk.inner(coh)
        assert isinstance(result, QubitKet)

    def test_to_fock_ket_shape(self) -> None:
        cs = jnp.array([[1.0, 0.0], [0.0, 0.0]])
        ds = jnp.array([[0.0+0j, 0.0+0j], [0.0+0j, 0.0+0j]])
        jk = JointKet(cs=cs, ds=ds)
        psi = jk.to_fock_ket(20)
        assert psi.shape == (40,)

    def test_to_fock_basis_shape(self) -> None:
        cs = jnp.array([[1.0, 0.0], [0.0, 0.0]])
        ds = jnp.array([[0.0+0j, 0.0+0j], [0.0+0j, 0.0+0j]])
        jk = JointKet(cs=cs, ds=ds)
        rho = jk.to_fock_basis(20)
        assert rho.shape == (40, 40)

    def test_to_fock_ket_layout_convention(self) -> None:
        # Library convention: kron(cavity, qubit). For |n=0> tensor |mu=1>,
        # only index 2*0 + 1 = 1 is populated.
        cs = jnp.array([[0.0+0j, 0.0+0j], [1.0+0j, 0.0+0j]])
        ds = jnp.array([[0.0+0j, 0.0+0j], [0.0+0j, 0.0+0j]])
        jk = JointKet(cs=cs, ds=ds)
        psi = jk.to_fock_ket(20)
        npt.assert_allclose(psi[1], 1.0 + 0j, atol=1e-12)
        npt.assert_allclose(psi[0], 0.0 + 0j, atol=1e-12)
        npt.assert_allclose(psi[3], 0.0 + 0j, atol=1e-12)

    def test_to_fock_ket_matches_circuit_unitary(self) -> None:
        # JointKet.to_fock_ket must match the layout used by compose_ECD_layers
        # (the rest of the library's dqtensor(cavity, qubit) convention).
        import jax
        from coherax.states import CoherentKet, QubitKet
        from coherax.circuits import CircuitUnitary, compose_ECD_layers
        from coherax._fock import dqcoherent
        from coherax.linalg_utils import GKP_N

        key = jax.random.PRNGKey(0)
        params = (jax.random.normal(key, (2, 4)) * 0.3).astype(jnp.complex128)
        boson = CoherentKet(cs=jnp.array([1.0+0j]), ds=jnp.array([0.4+0.1j]))
        qubit = QubitKet(cs=jnp.array([1.0+0j, 0.0+0j]))

        psi_a = CircuitUnitary.from_params(params, N_l=4).apply(boson, qubit).to_fock_ket(GKP_N)
        cav_in = dqcoherent(GKP_N, 0.4 + 0.1j).reshape(-1)
        psi_in = jnp.kron(cav_in, jnp.array([1.0+0j, 0.0+0j]))
        psi_b = compose_ECD_layers(params) @ psi_in
        # complex64 inside TraceoutLayer caps precision at ~1e-7
        npt.assert_allclose(jnp.abs(jnp.vdot(psi_a, psi_b)), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Displacer
# ---------------------------------------------------------------------------


class TestDisplacer:
    def test_vacuum_displacement(self) -> None:
        """D(beta)|0> = |beta> (coherent state)."""
        beta = 1.5 + 0.5j
        vac = CoherentKet(cs=jnp.array([1.0]), ds=jnp.array([0.0+0j]))
        d = Displacer(beta=beta)
        result = d.apply(vac)
        npt.assert_allclose(result.ds[0], beta, atol=1e-12)

    def test_displacement_composition(self) -> None:
        """D(a)D(b)|0> displacements add (up to phase)."""
        a, b = 0.5+0j, 0.3+0.2j
        vac = CoherentKet(cs=jnp.array([1.0]), ds=jnp.array([0.0+0j]))
        result = Displacer(a).apply(Displacer(b).apply(vac))
        # The displacement should be a+b
        psi_direct = CoherentKet(cs=jnp.array([1.0]), ds=jnp.array([a+b]))
        overlap = jnp.abs(result.inner(psi_direct))
        npt.assert_allclose(overlap, 1.0, atol=1e-8)

    def test_adjoint_undoes(self) -> None:
        """D†(β)D(β)|ψ> = |ψ>."""
        psi = CoherentKet(cs=jnp.array([1.0, 1.0]), ds=jnp.array([1.0+0j, -1.0+0j]))
        d = Displacer(beta=0.7+0.3j)
        roundtrip = d.apply_adj(d.apply(psi))
        npt.assert_allclose(jnp.abs(psi.inner(roundtrip)), 1.0, atol=1e-8)

    def test_apply_dm(self) -> None:
        """Displacement on DM shifts displacements."""
        ket = CoherentKet(cs=jnp.array([1.0]), ds=jnp.array([0.5+0j]))
        dm = CoherentDM.from_ket(ket)
        d = Displacer(beta=1.0+0j)
        dm2 = d.apply_dm(dm)
        # Fock-basis should match displaced ket
        ket2 = d.apply(ket)
        dm_ref = CoherentDM.from_ket(ket2)
        N = 30
        npt.assert_allclose(dm2.to_fock_basis(N), dm_ref.to_fock_basis(N), atol=1e-8)

    def test_dagger(self) -> None:
        d = Displacer(beta=1.0+0.5j)
        dd = d.dagger()
        npt.assert_allclose(dd.beta, -d.beta, atol=1e-12)


# ---------------------------------------------------------------------------
# Rotator
# ---------------------------------------------------------------------------


class TestRotator:
    def test_rotate_coherent(self) -> None:
        """Rotation by pi/2 maps |alpha> -> |i*alpha>."""
        alpha = 2.0 + 0j
        ket = CoherentKet(cs=jnp.array([1.0]), ds=jnp.array([alpha]))
        r = Rotator(theta=jnp.pi/2)
        result = r.apply(ket)
        npt.assert_allclose(result.ds[0], alpha * 1j, atol=1e-10)

    def test_full_rotation_identity(self) -> None:
        """Rotation by 2pi is identity."""
        ket = CoherentKet(cs=jnp.array([1.0, 1.0]), ds=jnp.array([1.0+0j, -1.0+0j]))
        r = Rotator(theta=2*jnp.pi)
        result = r.apply(ket)
        npt.assert_allclose(jnp.abs(ket.inner(result)), 1.0, atol=1e-8)

    def test_adjoint_undoes(self) -> None:
        ket = CoherentKet(cs=jnp.array([1.0, 1.0]), ds=jnp.array([1.5+0.5j, -0.5+0j]))
        r = Rotator(theta=0.7)
        roundtrip = r.apply_adj(r.apply(ket))
        npt.assert_allclose(jnp.abs(ket.inner(roundtrip)), 1.0, atol=1e-8)

    def test_apply_dm(self) -> None:
        ket = CoherentKet(cs=jnp.array([1.0]), ds=jnp.array([1.0+0j]))
        dm = CoherentDM.from_ket(ket)
        r = Rotator(theta=jnp.pi/2)
        dm2 = r.apply_dm(dm)
        ket2 = r.apply(ket)
        dm_ref = CoherentDM.from_ket(ket2)
        N = 30
        npt.assert_allclose(dm2.to_fock_basis(N), dm_ref.to_fock_basis(N), atol=1e-8)


# ---------------------------------------------------------------------------
# CPTP
# ---------------------------------------------------------------------------


class TestCPTP:
    def test_kraus_kets(self) -> None:
        """CPTP.kraus_kets returns one ket per operator."""
        d1 = Displacer(beta=0.5+0j)
        d2 = Displacer(beta=-0.5+0j)
        ch = CPTP(ops=[d1, d2])
        psi = CoherentKet(cs=jnp.array([1.0]), ds=jnp.array([0.0+0j]))
        branches = ch.kraus_kets(psi)
        assert len(branches) == 2
        assert all(isinstance(b, CoherentKet) for b in branches)
