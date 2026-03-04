"""
transpose_channel_claude.py

Transpose channel recovery for finite GKP codes under pure loss.
Implements Section 4.2 of UnifiedNotationGKP.pdf.

Core components:
- GKP logical state construction (coherent and Fock representations)
- SBS (Small-Big-Small) recovery baseline
- Transpose channel recovery (optimal among transpose-type maps)
- Entanglement fidelity computation for benchmarking
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np
from functools import partial
from jaxtyping import Array

from coherax.characteristic_jax_utils import (
    CoherentKet, BosonicSubspace,
    gkp_coherent_dm, coherent_overlap, aOmegab,
    sparse_eigh, dag,
    make_pureloss_fock, make_transpose_for_pureloss,
    apply_kraus_map, apply_kraus_map_nonorm,
    compose_channel_kraus, traceout_unitary,
    GKP_N, root2,
    dqtensor, dqdag, dqtrace,
    dqeye, dqnumber, dqdestroy, dqcreate,
    dqdisplace, dqcoherent,
    sigma_x, sigma_z,
    IN, I2,
)


# ============================================================
# GKP STATE CONSTRUCTION
# ============================================================

def build_gkp_states(Delta=0.3, N_trunc=3, lattice="square"):
    """
    Build GKP logical |0_L> and |1_L> as CoherentKet objects.

    Args:
        Delta: GKP envelope parameter (finite energy)
        N_trunc: truncation of lattice sums (uses (2*N_trunc+1)^2 coherent states)
        lattice: "square" or "rect"

    Returns:
        (logical_0, logical_1): CoherentKet objects
    """
    logical_0 = gkp_coherent_dm(mu=0, N_trunc=N_trunc, Delta=Delta, lattice=lattice)
    logical_1 = gkp_coherent_dm(mu=1, N_trunc=N_trunc, Delta=Delta, lattice=lattice)
    return logical_0, logical_1


def coherent_ket_to_fock(ck, N=GKP_N):
    """
    Convert a CoherentKet to a Fock-basis ket vector.

    Args:
        ck: CoherentKet object
        N: Fock space truncation

    Returns:
        (N, 1) array: Fock-basis ket
    """
    coherents = jax.vmap(lambda alpha: dqcoherent(N, alpha))(ck.ds)  # (A, N, 1)
    return jnp.einsum('ijk,i->jk', coherents, ck.cs)  # (N, 1)


# ============================================================
# SBS RECOVERY BASELINE
# ============================================================

def _cd_royer(beta, N=GKP_N):
    """
    Controlled Displacement in Royer convention (matches utils.py).
    CD(beta) = exp((beta a† - beta* a) ⊗ sigma_z / (2*sqrt(2)))
    """
    a_hat = dqdestroy(N)
    a_dag_hat = dqcreate(N)
    generator = dqtensor(
        beta * a_dag_hat - jnp.conj(beta) * a_hat,
        sigma_z
    ) / (2 * root2)
    return jla.expm(generator)


def _r_x(theta):
    """Qubit rotation R_x(theta) = exp(-i*theta*sigma_x/2)."""
    return jla.expm(-1j * theta * sigma_x / 2)


def build_sbs_unitary(direction, Delta=0.3, N=GKP_N):
    """
    Build SBS unitary for one stabilizer direction.

    U_SBS = CD_A_small @ (I x Rx†) @ CD_B @ (I x Rx) @ CD_A_small

    Args:
        direction: 0 or 1 (stabilizer index)
        Delta: GKP envelope parameter
        N: Fock space truncation

    Returns:
        (2N, 2N) unitary matrix
    """
    GKP_L = 2.0 * jnp.sqrt(jnp.pi)
    alpha_arr = GKP_L * jnp.array([0.0, 1.0], dtype=jnp.complex64)
    beta_arr = GKP_L * jnp.array([-1.0, 0.0], dtype=jnp.complex64)

    c_D = jnp.cosh(Delta**2)
    s_D = jnp.sinh(Delta**2)

    l_j = jnp.sqrt(jnp.abs(alpha_arr)**2 + jnp.abs(beta_arr)**2)
    theta_j = jnp.angle(alpha_arr + 1j * beta_arr)
    epsilon_j = s_D * 4 * jnp.pi / l_j

    j = direction
    eps_j = epsilon_j[j] * jnp.exp(1j * theta_j[j])
    big_disp = -1j * (alpha_arr[j] + 1j * beta_arr[j]) * c_D

    I_N = dqeye(N)
    cd_a_small = _cd_royer(eps_j / 2.0, N)
    cd_b = _cd_royer(big_disp, N)
    rx = _r_x(jnp.pi / 2.0)
    rx_dag = dqdag(rx)

    U = cd_a_small @ (dqtensor(I_N, rx_dag) @ (cd_b @ (dqtensor(I_N, rx) @ cd_a_small)))
    return U


def build_sbs_kraus(Delta=0.3, N=GKP_N):
    """
    Build full SBS recovery Kraus operators (both stabilizer directions).
    Direction 0 (position stabilizer) applied first, then direction 1 (momentum).

    The position stabilizer measurement inherently shifts the oscillator by
    half a lattice vector, so a correction displacement D(sqrt(pi/2)) is
    applied to all direction-0 Kraus operators.

    Note: SBS is designed for continuous stabilization, not one-shot recovery
    after a discrete loss event. For loss recovery, the transpose channel
    or optimized CD+R circuits are more appropriate.

    Returns:
        (4, N, N) array of Kraus operators
    """
    U0 = build_sbs_unitary(0, Delta=Delta, N=N)
    U1 = build_sbs_unitary(1, Delta=Delta, N=N)

    K0 = traceout_unitary(U0, N)  # (2, N, N)
    K1 = traceout_unitary(U1, N)  # (2, N, N)

    # Direction 0 needs correction: position stabilizer measurement shifts
    # by half lattice vector (sqrt(pi/2) in complex plane)
    half_lat = jnp.sqrt(jnp.pi / 2)
    corr0 = jnp.squeeze(dqdisplace(N, half_lat))
    K0_corr = jnp.array([corr0 @ K0[k] for k in range(K0.shape[0])])

    return compose_channel_kraus(K1, K0_corr)  # (4, N, N)


# ============================================================
# TRANSPOSE CHANNEL RECOVERY
# ============================================================

def fock_transpose_recovery(gamma, logical_0, logical_1, loss_rank=10, N=GKP_N):
    """
    Compute transpose channel recovery R_l = P_C E_l† E(P_C)^{-1/2} in Fock basis.

    Args:
        gamma: loss probability
        logical_0, logical_1: CoherentKet GKP logical states
        loss_rank: number of loss Kraus operators
        N: Fock space truncation

    Returns:
        transpose_ops: (loss_rank, N, N) recovery Kraus operators
        loss_ops: (loss_rank, N, N) loss Kraus operators
    """
    loss_ops = make_pureloss_fock(gamma, rank=loss_rank, N=N)
    transpose_ops = make_transpose_for_pureloss(loss_ops, logical_0, logical_1)
    return transpose_ops, loss_ops


def coherent_epc_check(gamma, logical_0, logical_1, N=GKP_N):
    """
    Validate E(P_C) by computing it in coherent basis and comparing to Fock.

    The coherent-basis computation uses:
      E(P_C) = sum_{a,b} p'_{ab} |d'_a><d'_b|
    where p'_{ab} = p_{ab} * <r*d_b|r*d_a> and d'_a = t*d_a.

    Returns:
        epc_coherent_fock: E(P_C) via coherent basis, synthesized to Fock
        epc_fock: E(P_C) via direct Fock computation
    """
    A0 = logical_0.cs.shape[0]
    A1 = logical_1.cs.shape[0]

    ds_all = jnp.concatenate([logical_0.ds, logical_1.ds])
    cs_0 = jnp.concatenate([logical_0.cs, jnp.zeros(A1)])
    cs_1 = jnp.concatenate([jnp.zeros(A0), logical_1.cs])

    # Codespace projector: p_{ab} = sum_mu c_{a,mu} c*_{b,mu}
    p_mat = jnp.outer(cs_0, jnp.conj(cs_0)) + jnp.outer(cs_1, jnp.conj(cs_1))

    t = jnp.sqrt(1 - gamma)
    r = jnp.sqrt(gamma)

    # Loss overlap: <r*d_b|r*d_a>
    rd = r * ds_all
    loss_inner = coherent_overlap(rd.reshape(1, -1), rd.reshape(-1, 1))

    p_prime = p_mat * loss_inner
    ds_prime = t * ds_all

    # Synthesize to Fock: E(P_C) = sum_{a,b} p'_{ab} |d'_a><d'_b|
    coherents_prime = jnp.squeeze(
        jax.vmap(lambda alpha: dqcoherent(N, alpha))(ds_prime)
    )  # (A, N)
    epc_coherent_fock = jnp.einsum(
        "ai,bj,ab->ij", coherents_prime, jnp.conj(coherents_prime), p_prime
    )

    # Fock reference (use default N to avoid JAX tracing issues with static args)
    P_fock = logical_0.to_fock_basis() + logical_1.to_fock_basis()
    loss_ops = make_pureloss_fock(gamma, rank=10, N=N)
    epc_fock = apply_kraus_map_nonorm(loss_ops, P_fock)

    return epc_coherent_fock, epc_fock


# ============================================================
# ENTANGLEMENT FIDELITY
# ============================================================

def entanglement_fidelity(recovery_ops, loss_ops, psi_0, psi_1):
    """
    Compute entanglement fidelity F_e for recovery channel R after loss E.

    F_e = (1/4) sum_{mu,nu} <mu_L| R(E(|mu_L><nu_L|)) |nu_L>

    Related to average fidelity: F_avg = (2*F_e + 1) / 3

    Args:
        recovery_ops: (K_R, N, N) recovery Kraus operators
        loss_ops: (K_E, N, N) loss Kraus operators
        psi_0: (N, 1) Fock ket for |0_L>
        psi_1: (N, 1) Fock ket for |1_L>

    Returns:
        F_e: entanglement fidelity (real scalar)
    """
    psi = [psi_0, psi_1]
    F_e = 0.0
    for mu in range(2):
        for nu in range(2):
            rho_mn = psi[mu] @ dqdag(psi[nu])
            after_loss = apply_kraus_map_nonorm(loss_ops, rho_mn)
            after_recovery = apply_kraus_map_nonorm(recovery_ops, after_loss)
            F_e += (dqdag(psi[mu]) @ after_recovery @ psi[nu]).squeeze()
    return jnp.real(F_e) / 4.0


def entanglement_fidelity_no_recovery(loss_ops, psi_0, psi_1):
    """Entanglement fidelity of loss channel alone (no recovery)."""
    psi = [psi_0, psi_1]
    F_e = 0.0
    for mu in range(2):
        for nu in range(2):
            rho_mn = psi[mu] @ dqdag(psi[nu])
            after_loss = apply_kraus_map_nonorm(loss_ops, rho_mn)
            F_e += (dqdag(psi[mu]) @ after_loss @ psi[nu]).squeeze()
    return jnp.real(F_e) / 4.0


# ============================================================
# COMPARISON DRIVER
# ============================================================

def compare_recoveries(gamma=0.05, Delta=0.3, N_trunc=3, loss_rank=10, N=GKP_N):
    """
    Compare SBS vs transpose channel entanglement fidelity.

    Returns:
        dict with fidelity values and operators
    """
    print(f"=== Recovery Comparison: gamma={gamma}, Delta={Delta} ===")

    # Build GKP states
    logical_0, logical_1 = build_gkp_states(Delta=Delta, N_trunc=N_trunc)
    psi_0 = coherent_ket_to_fock(logical_0, N)
    psi_1 = coherent_ket_to_fock(logical_1, N)

    # Normalize
    psi_0 = psi_0 / jnp.sqrt(jnp.real(dqdag(psi_0) @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(dqdag(psi_1) @ psi_1).squeeze())

    print(f"  |0_L> norm: {jnp.real(dqdag(psi_0) @ psi_0).squeeze():.6f}")
    print(f"  |1_L> norm: {jnp.real(dqdag(psi_1) @ psi_1).squeeze():.6f}")
    print(f"  |<0_L|1_L>|: {jnp.abs(dqdag(psi_0) @ psi_1).squeeze():.6f}")

    # Loss channel
    loss_ops = make_pureloss_fock(gamma, rank=loss_rank, N=N)

    # No recovery baseline
    Fe_none = entanglement_fidelity_no_recovery(loss_ops, psi_0, psi_1)
    print(f"  F_e (no recovery): {Fe_none:.6f}")

    # SBS recovery (note: SBS is a stabilization protocol, not designed for
    # one-shot loss recovery; included as reference only)
    sbs_ops = build_sbs_kraus(Delta=Delta, N=N)
    Fe_sbs = entanglement_fidelity(sbs_ops, loss_ops, psi_0, psi_1)
    print(f"  F_e (SBS):         {Fe_sbs:.6f}  (stabilization protocol, not loss recovery)")

    # Transpose channel recovery
    transpose_ops = make_transpose_for_pureloss(loss_ops, logical_0, logical_1)
    Fe_transpose = entanglement_fidelity(transpose_ops, loss_ops, psi_0, psi_1)
    print(f"  F_e (transpose):   {Fe_transpose:.6f}")

    # Coherent basis validation
    epc_coh, epc_fock = coherent_epc_check(gamma, logical_0, logical_1, N=N)
    epc_diff = jnp.max(jnp.abs(epc_coh - epc_fock))
    print(f"  E(P_C) coh vs Fock diff: {epc_diff:.2e}")

    return {
        'gamma': gamma,
        'Delta': Delta,
        'Fe_none': float(Fe_none),
        'Fe_sbs': float(Fe_sbs),
        'Fe_transpose': float(Fe_transpose),
        'epc_diff': float(epc_diff),
        'sbs_ops': sbs_ops,
        'transpose_ops': transpose_ops,
        'loss_ops': loss_ops,
        'psi_0': psi_0,
        'psi_1': psi_1,
        'logical_0': logical_0,
        'logical_1': logical_1,
    }


if __name__ == "__main__":
    results = []
    for gamma in [0.01, 0.03, 0.05, 0.07, 0.1]:
        res = compare_recoveries(gamma=gamma, Delta=0.3, N_trunc=3, loss_rank=10)
        results.append(res)

    print("\n=== Summary ===")
    print(f"{'gamma':>8s} | {'F_e(none)':>10s} | {'F_e(SBS)':>10s} | {'F_e(transpose)':>14s}")
    print("-" * 50)
    for r in results:
        print(
            f"{r['gamma']:8.3f} | {r['Fe_none']:10.6f} | "
            f"{r['Fe_sbs']:10.6f} | {r['Fe_transpose']:14.6f}"
        )
