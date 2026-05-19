"""Tests for the floating-basis additions.

Covers:
- ``beamsplit_full`` (typed + raw kernel) against pure-loss Kraus channel.
- Encoder isometry: :math:`C^\\dagger G C = I_D`.
- Coherent-basis :math:`I_c` agrees with Fock-basis Kraus :math:`I_c` (the
  staging notebook's central sanity check).
- ``entanglement_fidelity_pureloss`` is in :math:`[0, 1]`.
- ``nbar_logical`` against a direct expectation-value computation.
- ``separation_penalty`` zero on well-separated points, positive on close ones.
- End-to-end smoke test of both optimizers.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpy.testing as npt
import pytest

from coherax import (
    CoherentKet,
    apply_kraus_map_nonorm,
    beamsplit_full,
    coherent_information_pureloss,
    complex_normal,
    dqcoherent,
    encode_logical_ket,
    encode_logical_kets,
    entanglement_fidelity_pureloss,
    init_separated_d,
    make_pureloss_fock,
    nbar_logical,
    optimize_Fe_floating,
    optimize_Ic_floating,
    separation_penalty,
)
from coherax.linalg_utils import coherent_overlap, dag
from coherax.states import _beamsplit_full_arrays, unitary_encoding_map


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_encoding(seed: int = 0, N_C: int = 6, D: int = 2) -> tuple[jnp.ndarray, jnp.ndarray]:
    key = jr.PRNGKey(seed)
    k1, k2 = jr.split(key)
    X = (0.5 * complex_normal(k1, (N_C, D))).astype(jnp.complex128)
    d = init_separated_d(k2, N_C, min_sep=1.0)
    return X, d


# ---------------------------------------------------------------------------
# beamsplit_full
# ---------------------------------------------------------------------------


class TestBeamsplit:
    def test_typed_matches_raw(self) -> None:
        key = jr.PRNGKey(0)
        k1, k2, k3 = jr.split(key, 3)
        D, A, N_E = 2, 4, 3
        cs_stack = (0.5 * complex_normal(k1, (D, A))).astype(jnp.complex128)
        d = (jr.normal(k2, (A,)) + 1j * jr.normal(k3, (A,))).astype(jnp.complex128)
        env = CoherentKet(
            cs=jnp.array([1.0 + 0j, 0.5 - 0.2j, 0.3 + 0.1j]),
            ds=jnp.array([0.0 + 0j, 1.0 + 0j, -1.0 + 0j]),
        )
        # Build CoherentKets (these get normalized — match alpha after)
        logical_kets = [CoherentKet(cs=cs_stack[mu], ds=d) for mu in range(D)]
        alpha_norm = jnp.stack([k.cs for k in logical_kets])
        rho_typed, d_typed = beamsplit_full(logical_kets, env, 0.7)
        rho_raw, d_raw = _beamsplit_full_arrays(alpha_norm, d, env.cs, env.ds, jnp.asarray(0.7))
        npt.assert_allclose(rho_typed, rho_raw, atol=1e-14)
        npt.assert_allclose(d_typed, d_raw, atol=1e-14)

    def test_pureloss_on_coherent_state_matches_kraus(self) -> None:
        # Single coherent state through beamsplitter with vacuum env equals
        # the Kraus pure-loss channel applied to |alpha><alpha| in Fock basis.
        alpha = 1.5 + 0.5j
        gamma = 0.3
        eta = 1.0 - gamma
        N_fock = 50
        ket = CoherentKet(cs=jnp.array([1.0 + 0j]), ds=jnp.array([alpha]))
        env = CoherentKet(cs=jnp.array([1.0 + 0j]), ds=jnp.array([0.0 + 0j]))
        rho_coh, d_out = beamsplit_full([ket], env, eta)
        c00 = rho_coh[0, 0, 0, 0]
        coh_vec = dqcoherent(N_fock, d_out[0]).reshape(-1)
        rho_beam = c00 * jnp.outer(coh_vec, jnp.conj(coh_vec))
        rho_beam = rho_beam / jnp.trace(rho_beam)

        psi_fock = dqcoherent(N_fock, alpha).reshape(-1)
        rho_in = jnp.outer(psi_fock, jnp.conj(psi_fock))
        ops = make_pureloss_fock(gamma, 20, N_fock)
        rho_kraus = apply_kraus_map_nonorm(ops, rho_in)
        rho_kraus = rho_kraus / jnp.trace(rho_kraus)
        npt.assert_allclose(rho_beam, rho_kraus, atol=1e-10)

    def test_eta_one_is_identity_on_displacements(self) -> None:
        # With eta=1 (no loss) and vacuum environment, d_out should equal d.
        ket = CoherentKet(cs=jnp.array([1.0 + 0j, 0.5 + 0j]),
                          ds=jnp.array([0.5 + 0j, -0.5 + 0j]))
        env = CoherentKet(cs=jnp.array([1.0 + 0j]), ds=jnp.array([0.0 + 0j]))
        _, d_out = beamsplit_full([ket], env, 1.0)
        # d_out has shape (A * N_E,) = (2,); should match ket.ds
        npt.assert_allclose(jnp.sort(jnp.real(d_out)),
                            jnp.sort(jnp.real(ket.ds)), atol=1e-12)

    def test_mismatched_ds_raises(self) -> None:
        env = CoherentKet(cs=jnp.array([1.0 + 0j]), ds=jnp.array([0.0 + 0j]))
        with pytest.raises(ValueError):
            beamsplit_full(
                [
                    CoherentKet(cs=jnp.array([1.0]), ds=jnp.array([0.0 + 0j])),
                    CoherentKet(cs=jnp.array([1.0, 0.5]),
                                ds=jnp.array([0.0 + 0j, 1.0 + 0j])),
                ],
                env,
                0.9,
            )


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class TestEncoder:
    def test_isometry(self) -> None:
        # C^dag G C = I_D for the encoder coefficient matrix.
        X, d = _random_encoding(seed=42)
        A = d.shape[0]
        psi_0 = jnp.array([1.0 + 0j, 0.0 + 0j])
        psi_1 = jnp.array([0.0 + 0j, 1.0 + 0j])
        c0 = unitary_encoding_map(X, d, psi_0)
        c1 = unitary_encoding_map(X, d, psi_1)
        C = jnp.stack([c0, c1], axis=1)
        G = coherent_overlap(d.reshape(A, 1), d.reshape(1, A))
        iso = dag(C) @ G @ C
        npt.assert_allclose(iso, jnp.eye(2), atol=1e-10)

    def test_encode_logical_ket_returns_coherentket(self) -> None:
        X, d = _random_encoding(seed=1)
        ket = encode_logical_ket(X, d, mu=0)
        assert isinstance(ket, CoherentKet)
        # Already unit-norm by construction
        npt.assert_allclose(jnp.real(ket.inner(ket)), 1.0, atol=1e-10)

    def test_encode_logical_kets_orthonormal(self) -> None:
        X, d = _random_encoding(seed=2)
        kets = encode_logical_kets(X, d)
        assert len(kets) == 2
        npt.assert_allclose(jnp.real(kets[0].inner(kets[0])), 1.0, atol=1e-10)
        npt.assert_allclose(jnp.real(kets[1].inner(kets[1])), 1.0, atol=1e-10)
        npt.assert_allclose(jnp.abs(kets[0].inner(kets[1])), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# F_e and I_c
# ---------------------------------------------------------------------------


class TestFidelityFunctions:
    def test_Fe_in_unit_interval(self) -> None:
        X, d = _random_encoding(seed=10)
        Z = (0.5 * complex_normal(jr.PRNGKey(99), (10, 2, d.shape[0]))).astype(
            jnp.complex128
        )
        Fe = float(entanglement_fidelity_pureloss(X, d, Z, 0.05))
        assert 0.0 <= Fe <= 1.0 + 1e-10

    def test_Ic_coherent_matches_fock_kraus(self) -> None:
        # The central correctness check from the staging notebook:
        # coherent-basis I_c == Fock-basis Kraus-channel I_c.
        X, d = _random_encoding(seed=1234)
        psi_0 = jnp.array([1.0 + 0j, 0.0 + 0j])
        psi_1 = jnp.array([0.0 + 0j, 1.0 + 0j])
        c0 = unitary_encoding_map(X, d, psi_0)
        c1 = unitary_encoding_map(X, d, psi_1)
        N_fock = 70

        def _ic_fock(gamma: float) -> float:
            coh_basis = jnp.stack(
                [dqcoherent(N_fock, da).reshape(-1) for da in np.asarray(d)]
            )
            psi0 = (c0[:, None] * coh_basis).sum(0)
            psi0 = psi0 / jnp.linalg.norm(psi0)
            psi1 = (c1[:, None] * coh_basis).sum(0)
            psi1 = psi1 / jnp.linalg.norm(psi1)
            ops = make_pureloss_fock(gamma, 20, N_fock)
            v0 = jnp.einsum("knm,m->kn", ops, psi0)
            v1 = jnp.einsum("knm,m->kn", ops, psi1)
            b00 = jnp.einsum("kn,km->nm", v0, jnp.conj(v0))
            b11 = jnp.einsum("kn,km->nm", v1, jnp.conj(v1))
            b01 = jnp.einsum("kn,km->nm", v0, jnp.conj(v1))
            b10 = jnp.einsum("kn,km->nm", v1, jnp.conj(v0))
            rho_B = 0.5 * (b00 + b11)
            rho_RB = 0.5 * jnp.block([[b00, b01], [b10, b11]])

            def ent(rho):
                w = jnp.real(jnp.linalg.eigvalsh(rho))
                return float(
                    -jnp.sum(
                        jnp.where(w > 1e-15, w * jnp.log(jnp.maximum(w, 1e-30)), 0.0)
                    )
                )
            return (ent(rho_B) - ent(rho_RB)) / np.log(2)

        for gamma in (0.05, 0.10, 0.20):
            ic_coh = float(coherent_information_pureloss(X, d, gamma))
            ic_fock = _ic_fock(gamma)
            npt.assert_allclose(ic_coh, ic_fock, atol=1e-8)

    def test_nbar_against_direct_expectation(self) -> None:
        # nbar = (1/D) sum_mu <psi_mu|n_hat|psi_mu> using a Fock-basis
        # reference (truncated, but large enough to converge).
        X, d = _random_encoding(seed=7)
        nbar_coh = float(nbar_logical(X, d))

        N_fock = 80
        coh_basis = jnp.stack(
            [dqcoherent(N_fock, da).reshape(-1) for da in np.asarray(d)]
        )
        n_op = jnp.diag(jnp.arange(N_fock, dtype=jnp.float64).astype(jnp.complex128))
        D = X.shape[1]
        total = 0.0
        for mu in range(D):
            psi_mu_logical = jnp.zeros(D, dtype=jnp.complex128).at[mu].set(1.0)
            cmu = unitary_encoding_map(X, d, psi_mu_logical)
            psi = (cmu[:, None] * coh_basis).sum(0)
            psi = psi / jnp.linalg.norm(psi)
            total = total + float(jnp.real(jnp.conj(psi) @ n_op @ psi))
        nbar_fock = total / D
        npt.assert_allclose(nbar_coh, nbar_fock, atol=1e-6)


# ---------------------------------------------------------------------------
# Separation penalty
# ---------------------------------------------------------------------------


class TestSeparationPenalty:
    def test_zero_for_well_separated(self) -> None:
        # min_sep=1.0; distances all >= 2.
        d = jnp.array([0.0, 2.0, 4.0, -2.0], dtype=jnp.complex128)
        assert float(separation_penalty(d, min_sep=1.0)) == 0.0

    def test_positive_for_close_points(self) -> None:
        d = jnp.array([0.0, 0.1, 5.0], dtype=jnp.complex128)
        assert float(separation_penalty(d, min_sep=1.0)) > 0.0

    def test_init_separated_d_separation(self) -> None:
        d = init_separated_d(jr.PRNGKey(5), N_C=8, min_sep=1.0)
        diffs = jnp.abs(d.reshape(-1, 1) - d.reshape(1, -1))
        # Ignore the zero diagonal
        diffs = diffs + 1e6 * jnp.eye(8)
        # Initialization perturbation may slightly violate; allow 50% margin
        assert float(jnp.min(diffs)) > 0.5


# ---------------------------------------------------------------------------
# Optimizer smoke tests
# ---------------------------------------------------------------------------


class TestOptimizers:
    def test_optimize_Fe_smoke(self) -> None:
        res = optimize_Fe_floating(
            gamma=0.10, N_C=6, N_D=4, restarts=2,
            steps_p1=200, steps_p2=100, seed=0, verbose=False,
        )
        assert res["n_valid"] >= 1
        assert 0.0 <= res["Fe"] <= 1.0
        assert res["X"].shape == (6, 2)
        assert res["d"].shape == (6,)
        assert res["Z"].shape == (4, 2, 6)
        # Sanity: at gamma=0.10 with this tiny budget, F_e should clear 0.85.
        assert res["Fe"] > 0.85, f"unusually low F_e: {res['Fe']}"

    def test_optimize_Ic_smoke(self) -> None:
        res = optimize_Ic_floating(
            gamma=0.10, N_C=6, restarts=2,
            steps_p1=200, steps_p2=100, seed=0, verbose=False,
        )
        assert res["n_valid"] >= 1
        assert -0.01 <= res["Ic"] <= 1.01
        assert res["X"].shape == (6, 2)
        assert res["d"].shape == (6,)
        # Sanity: at gamma=0.10, I_c should clear 0.9 qubits.
        assert res["Ic"] > 0.9, f"unusually low I_c: {res['Ic']}"
