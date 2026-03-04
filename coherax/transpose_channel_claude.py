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

from coherax import (
    CoherentKet,
    gkp_coherent_dm,
    make_pureloss_fock, make_transpose_for_pureloss,
    apply_kraus_map_nonorm,
    GKP_N,
    dqdag, dqcoherent,
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


if __name__ == "__main__":
    for gamma in [0.01, 0.03, 0.05, 0.07, 0.1]:
        logical_0, logical_1 = build_gkp_states(Delta=0.3, N_trunc=3)
        psi_0 = coherent_ket_to_fock(logical_0)
        psi_1 = coherent_ket_to_fock(logical_1)
        psi_0 = psi_0 / jnp.sqrt(jnp.real(dqdag(psi_0) @ psi_0).squeeze())
        psi_1 = psi_1 / jnp.sqrt(jnp.real(dqdag(psi_1) @ psi_1).squeeze())

        loss_ops = make_pureloss_fock(gamma, rank=10)
        Fe_none = entanglement_fidelity_no_recovery(loss_ops, psi_0, psi_1)
        transpose_ops = make_transpose_for_pureloss(loss_ops, logical_0, logical_1)
        Fe_trans = entanglement_fidelity(transpose_ops, loss_ops, psi_0, psi_1)
        print(f"gamma={gamma:.2f}: F_e(none)={Fe_none:.6f}, F_e(transpose)={Fe_trans:.6f}")
