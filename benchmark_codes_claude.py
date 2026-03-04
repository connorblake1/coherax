"""
Benchmark bosonic codes (cat, binomial, GKP) in coherent basis representation.
Provides code generators, mean photon number calculators, and fidelity evaluators
for comparison with floating-basis optimized codes.
"""

import sys
sys.path.insert(0, "../FiniteGKP/gkp_utils")

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from characteristic_jax_utils import (
    CoherentKet,
    CoherentDM,
    BosonicSubspace,
    coherent_overlap,
    aOmegab,
    gkp_coherent_dm,
    make_pureloss_fock,
    make_transpose_for_pureloss,
    apply_kraus_map,
    apply_kraus_map_nonorm,
    dqtrace,
    dqdag,
    dqcoherent,
    dqnumber,
    dqfock_dm,
    GKP_N,
)

jax.config.update("jax_enable_x64", True)


# ── Cat codes ────────────────────────────────────────────────────────────────

def cat_2leg(alpha: complex) -> tuple[CoherentKet, CoherentKet]:
    """2-legged cat code: |0_L> ~ |α⟩+|-α⟩, |1_L> ~ |α⟩-|-α⟩."""
    a = jnp.array(alpha, dtype=jnp.complex128)
    ds = jnp.array([a, -a])
    cs0 = jnp.array([1.0 + 0j, 1.0 + 0j])
    cs1 = jnp.array([1.0 + 0j, -1.0 + 0j])
    return CoherentKet(cs0, ds), CoherentKet(cs1, ds)


def cat_4leg(alpha: complex) -> tuple[CoherentKet, CoherentKet]:
    """4-legged cat code: |0_L> ~ |α⟩+|-α⟩+|iα⟩+|-iα⟩, |1_L> ~ |α⟩+|-α⟩-|iα⟩-|-iα⟩."""
    a = jnp.array(alpha, dtype=jnp.complex128)
    ds = jnp.array([a, -a, 1j * a, -1j * a])
    cs0 = jnp.array([1.0 + 0j, 1.0 + 0j, 1.0 + 0j, 1.0 + 0j])
    cs1 = jnp.array([1.0 + 0j, 1.0 + 0j, -1.0 + 0j, -1.0 + 0j])
    return CoherentKet(cs0, ds), CoherentKet(cs1, ds)


# ── Binomial codes ───────────────────────────────────────────────────────────

def binomial_order1() -> tuple[CoherentKet, CoherentKet]:
    """Order-1 binomial code: |0_L⟩ = (|0⟩+|4⟩)/√2, |1_L⟩ = |2⟩.
    Converts from Fock to coherent basis via inner products."""
    N = GKP_N
    # Fock-basis states
    psi0_fock = (dqcoherent(N, 0.0 + 0j).flatten() * 0)  # zero vector
    psi0_fock = psi0_fock.at[0].set(1.0 / jnp.sqrt(2.0))
    psi0_fock = psi0_fock.at[4].set(1.0 / jnp.sqrt(2.0))

    psi1_fock = psi0_fock * 0
    psi1_fock = psi1_fock.at[2].set(1.0 + 0j)

    # Use a grid of coherent states to represent these
    # The binomial code lives in a small Fock subspace, so moderate alpha suffices
    alphas = []
    for re in np.linspace(-3.0, 3.0, 7):
        for im in np.linspace(-3.0, 3.0, 7):
            alphas.append(re + 1j * im)
    alphas = jnp.array(alphas, dtype=jnp.complex128)

    # Compute coherent-basis coefficients via overlap: c_i = <alpha_i|psi>
    coherents = jax.vmap(lambda a: dqcoherent(N, a).flatten())(alphas)  # (M, N)
    cs0 = jnp.einsum("mi,i->m", jnp.conj(coherents), psi0_fock)
    cs1 = jnp.einsum("mi,i->m", jnp.conj(coherents), psi1_fock)

    return CoherentKet(cs0, alphas), CoherentKet(cs1, alphas)


# ── GKP codes ────────────────────────────────────────────────────────────────

def gkp_square(Delta: float, N_trunc: int = 5) -> tuple[CoherentKet, CoherentKet]:
    """Finite-energy square GKP code."""
    log0 = gkp_coherent_dm(mu=0, N_trunc=N_trunc, Delta=Delta, lattice="square")
    log1 = gkp_coherent_dm(mu=1, N_trunc=N_trunc, Delta=Delta, lattice="square")
    return log0, log1


def gkp_hex(Delta: float, N_trunc: int = 5) -> tuple[CoherentKet, CoherentKet]:
    """Finite-energy hexagonal GKP code.

    Lattice vectors: α = √(π/√3), β = √(π/√3)·e^{iπ/3}
    giving a 60° lattice with symplectic area π/2 per unit cell.
    The hexagonal lattice is optimal for single-mode pure loss.
    """
    a_mag = jnp.sqrt(jnp.pi / jnp.sqrt(3.0))
    GKP_alpha = a_mag  # real
    GKP_beta = a_mag * jnp.exp(1.0j * jnp.pi / 3.0)  # 60° angle

    states = []
    for mu in range(2):
        cs = []
        ds = []
        for k in range(-N_trunc, N_trunc + 1):
            for l in range(-N_trunc, N_trunc + 1):
                disp = (2 * k + mu) * GKP_alpha + l * GKP_beta
                cs.append(
                    jnp.exp(
                        -1.0j * jnp.pi * (k * l + l * mu / 2.0)
                        - (Delta ** 2) * jnp.abs(disp) ** 2
                    )
                )
                ds.append(disp)
        states.append(CoherentKet(cs=jnp.array(cs), ds=jnp.array(ds)))
    return states[0], states[1]


# ── Mean photon number ───────────────────────────────────────────────────────

def nbar_coherent_ket(state: CoherentKet) -> float:
    """Compute ⟨n̂⟩ for a CoherentKet.
    ⟨n̂⟩ = Σ_{ij} c_i* c_j ⟨d_i|n̂|d_j⟩ where ⟨α|n̂|β⟩ = α* β ⟨α|β⟩."""
    cs = state.cs
    ds = state.ds
    A = ds.shape[0]
    da = ds.reshape(A, 1)
    db = ds.reshape(1, A)
    ca = cs.reshape(A, 1)
    cb = cs.reshape(1, A)
    G = coherent_overlap(da, db)  # (A, A)
    # <alpha|n_hat|beta> = conj(alpha)*beta * <alpha|beta>
    nhat_matrix = jnp.conj(da) * db * G
    return jnp.real(jnp.sum(jnp.conj(ca) * cb * nhat_matrix))


def nbar_code(log0: CoherentKet, log1: CoherentKet) -> float:
    """Mean photon number of the maximally mixed logical state (|0_L⟩⟨0_L| + |1_L⟩⟨1_L|)/2."""
    return 0.5 * (nbar_coherent_ket(log0) + nbar_coherent_ket(log1))


# ── Loss channel fidelity ────────────────────────────────────────────────────

def entanglement_fidelity_fock(
    log0: CoherentKet,
    log1: CoherentKet,
    gamma: float,
    loss_rank: int = 20,
    use_transpose: bool = True,
) -> float:
    """Compute entanglement fidelity of a code under pure loss.

    Uses Fock-basis computation with transpose (Petz) recovery channel.

    Args:
        log0, log1: Logical code states as CoherentKet
        gamma: Loss rate (1 - transmissivity)
        loss_rank: Number of Kraus operators for loss channel
        use_transpose: If True, use Petz transpose recovery

    Returns:
        Entanglement fidelity F_e
    """
    N = GKP_N
    rho0 = log0.to_fock_basis()
    rho1 = log1.to_fock_basis()

    loss_ops = make_pureloss_fock(gamma, loss_rank, N)

    if use_transpose:
        recovery_ops = make_transpose_for_pureloss(loss_ops, log0, log1)
        # F_e = (1/D) Σ_μ ⟨μ_L| R∘L(|μ_L⟩⟨μ_L|) |μ_L⟩
        # For D=2: F_e = (1/2)(F_0 + F_1) where F_μ = Tr(ρ_μ R∘L(ρ_μ))
        # Plus off-diagonal: F_e = (1/D²) Σ_{μν} ⟨μ|(R∘L)(|μ⟩⟨ν|)|ν⟩
        # Compose loss then recovery
        combined_ops = compose_kraus(recovery_ops, loss_ops)
        # Entanglement fidelity
        F_e = _entanglement_fidelity_from_kraus(combined_ops, rho0, rho1)
    else:
        # Just loss channel, no recovery
        F_e = _entanglement_fidelity_from_kraus(loss_ops, rho0, rho1)

    return float(jnp.real(F_e))


def compose_kraus(ch1, ch2):
    """Compose two Kraus channels: ch1 ∘ ch2."""
    K1, K2 = ch1.shape[0], ch2.shape[0]
    N1, N2 = ch1.shape[1], ch2.shape[2]
    ops = jnp.zeros((K1 * K2, N1, N2), dtype=ch1.dtype)
    for i in range(K1):
        for j in range(K2):
            ops = ops.at[i * K2 + j].set(ch1[i] @ ch2[j])
    return ops


def _entanglement_fidelity_from_kraus(ops, rho0, rho1):
    """Compute entanglement fidelity F_e = (1/D²) Σ_k |Σ_μ ⟨μ_L|K_k|μ_L⟩|²
    for D=2 with logical states rho0=|0_L⟩⟨0_L|, rho1=|1_L⟩⟨1_L|."""
    # Extract leading eigenvector of each density matrix
    w0, v0 = jnp.linalg.eigh(rho0)
    psi0 = v0[:, -1]  # eigenvector with largest eigenvalue
    w1, v1 = jnp.linalg.eigh(rho1)
    psi1 = v1[:, -1]

    D = 2
    # F_e = (1/D²) Σ_k |Σ_μ ⟨μ_L|K_k|μ_L⟩|²
    # For each Kraus operator, compute trace in logical subspace
    def kraus_trace(K):
        return jnp.dot(jnp.conj(psi0), K @ psi0) + jnp.dot(jnp.conj(psi1), K @ psi1)

    traces = jax.vmap(kraus_trace)(ops)
    F_e = jnp.sum(jnp.abs(traces) ** 2) / (D ** 2)
    return F_e


# ── Match n̄ for fair comparison ──────────────────────────────────────────────

def find_cat_alpha_for_nbar(target_nbar: float, legs: int = 2) -> float:
    """Find cat code alpha that gives a target mean photon number.
    For large |α|: n̄ ≈ |α|² for both 2-leg and 4-leg cats."""
    # Simple binary search
    lo, hi = 0.1, 10.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if legs == 2:
            l0, l1 = cat_2leg(mid)
        else:
            l0, l1 = cat_4leg(mid)
        nb = float(nbar_code(l0, l1))
        if nb < target_nbar:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def find_gkp_delta_for_nbar(target_nbar: float, N_trunc: int = 5, lattice: str = "hex") -> float:
    """Find GKP Delta parameter that gives a target mean photon number."""
    gkp_fn = gkp_hex if lattice == "hex" else gkp_square
    lo, hi = 0.1, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        l0, l1 = gkp_fn(mid, N_trunc)
        nb = float(nbar_code(l0, l1))
        if nb > target_nbar:  # larger Delta = more squeezing = lower nbar
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ── Convenience: compute all benchmarks at given gamma/nbar ──────────────────

def benchmark_all(gamma: float, target_nbar: float, loss_rank: int = 20):
    """Compute entanglement fidelity for all benchmark codes at given gamma and n̄.

    Returns dict with keys: cat2, cat4, binomial, gkp, and their n̄ values.
    """
    results = {}

    # Cat codes matched to nbar
    alpha2 = find_cat_alpha_for_nbar(target_nbar, legs=2)
    l0_c2, l1_c2 = cat_2leg(alpha2)
    results["cat2_nbar"] = float(nbar_code(l0_c2, l1_c2))
    results["cat2_Fe"] = entanglement_fidelity_fock(l0_c2, l1_c2, gamma, loss_rank)
    results["cat2_alpha"] = float(alpha2)

    alpha4 = find_cat_alpha_for_nbar(target_nbar, legs=4)
    l0_c4, l1_c4 = cat_4leg(alpha4)
    results["cat4_nbar"] = float(nbar_code(l0_c4, l1_c4))
    results["cat4_Fe"] = entanglement_fidelity_fock(l0_c4, l1_c4, gamma, loss_rank)
    results["cat4_alpha"] = float(alpha4)

    # Binomial code (fixed n̄)
    l0_b, l1_b = binomial_order1()
    results["binomial_nbar"] = float(nbar_code(l0_b, l1_b))
    results["binomial_Fe"] = entanglement_fidelity_fock(l0_b, l1_b, gamma, loss_rank)

    # GKP (hexagonal) matched to nbar
    delta = find_gkp_delta_for_nbar(target_nbar, lattice="hex")
    l0_g, l1_g = gkp_hex(delta)
    results["gkp_nbar"] = float(nbar_code(l0_g, l1_g))
    results["gkp_Fe"] = entanglement_fidelity_fock(l0_g, l1_g, gamma, loss_rank)
    results["gkp_delta"] = float(delta)

    return results


if __name__ == "__main__":
    print("Testing benchmark codes...")

    # Test cat codes
    l0, l1 = cat_2leg(2.0)
    print(f"2-leg cat(α=2): n̄ = {nbar_code(l0, l1):.3f}")

    l0, l1 = cat_4leg(2.0)
    print(f"4-leg cat(α=2): n̄ = {nbar_code(l0, l1):.3f}")

    # Test binomial
    l0, l1 = binomial_order1()
    print(f"Binomial order-1: n̄ = {nbar_code(l0, l1):.3f}")

    # Test GKP (hexagonal)
    l0, l1 = gkp_hex(0.3)
    print(f"Hex GKP(Δ=0.3): n̄ = {nbar_code(l0, l1):.3f}")

    # Test GKP (square, for reference)
    l0, l1 = gkp_square(0.3)
    print(f"Sq  GKP(Δ=0.3): n̄ = {nbar_code(l0, l1):.3f}")

    # Test fidelity
    gamma = 0.05
    l0, l1 = cat_2leg(2.0)
    Fe = entanglement_fidelity_fock(l0, l1, gamma)
    print(f"2-leg cat(α=2), γ={gamma}: F_e = {Fe:.6f}")
