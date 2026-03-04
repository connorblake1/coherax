"""
coherent_transpose_claude.py

Coherent-basis transpose channel construction for finite GKP codes under pure loss.
Implements Section 4.2 of UnifiedNotationGKP.pdf WITHOUT any Fock-basis projection.

Pipeline:
  1. Build GKP logical states as CoherentKet
  2. Construct codespace projector P_C
  3. Compute E(P_C)^{-1/2} in orthonormal basis of V'
  4. Build Q tensor (Choi matrix of R^T) directly in ortho bases
  5. Eigendecompose Q to get top-K Kraus operators R_k
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jaxtyping import Array

from coherax.characteristic_jax_utils import (
    CoherentKet,
    BosonicSubspace,
    gkp_coherent_dm,
    coherent_overlap,
    aOmegab,
    sparse_eigh,
    sparse_tensor_eigh,
    dag,
    GKP_N,
    dqdag,
    dqtrace,
    dqcoherent,
    make_pureloss_fock,
    make_transpose_for_pureloss,
    apply_kraus_map_nonorm,
)


# ============================================================
# STEP 1: GKP STATES AND CODESPACE PROJECTOR
# ============================================================

def build_gkp_coherent_code(Delta=0.3, N_trunc=3, lattice="square", lam=None):
    """
    Build GKP code in coherent basis.

    For the square lattice (default), the logical states are well-separated with
    |GKP_alpha| = sqrt(pi/2) ~ 1.25, giving proper orthogonality for Delta=0.3.

    Returns:
        logical_0, logical_1: CoherentKet objects
        ds_all: (B,) all coherent state positions (union of both logicals)
        cs_0, cs_1: (B,) zero-padded coefficient vectors
        p_mat: (B, B) codespace projector coefficient matrix
    """
    kwargs = {'N_trunc': N_trunc, 'Delta': Delta, 'lattice': lattice}
    if lam is not None:
        kwargs['lam'] = lam
    logical_0 = gkp_coherent_dm(mu=0, **kwargs)
    logical_1 = gkp_coherent_dm(mu=1, **kwargs)

    A0 = logical_0.cs.shape[0]
    A1 = logical_1.cs.shape[0]
    B = A0 + A1

    ds_all = jnp.concatenate([logical_0.ds, logical_1.ds])
    cs_0 = jnp.concatenate([logical_0.cs, jnp.zeros(A1, dtype=jnp.complex64)])
    cs_1 = jnp.concatenate([jnp.zeros(A0, dtype=jnp.complex64), logical_1.cs])

    # Codespace projector: P_C = |0_L><0_L| + |1_L><1_L|
    # As coefficient matrix: p_{ab} = c_{a,0}*conj(c_{b,0}) + c_{a,1}*conj(c_{b,1})
    p_mat = (jnp.outer(cs_0, jnp.conj(cs_0))
             + jnp.outer(cs_1, jnp.conj(cs_1)))

    return logical_0, logical_1, ds_all, cs_0, cs_1, p_mat


# ============================================================
# STEP 2: Q TENSOR (ORTHO-FIRST APPROACH)
# ============================================================

def build_Q_tensor(p_mat, ds_all, gamma, eps=1e-6, verbose=False):
    """
    Build the Q tensor (Choi matrix of R^T) directly in orthonormal bases.

    Uses the ortho-first approach: all matrix elements computed in orthonormal
    bases, avoiding coefficient-vs-matrix-element ambiguity.

    The transpose channel R^T has continuous Kraus operators:
        R_z = P_C E_z^dag E(P_C)^{-1/2}

    where E_z is the loss channel Kraus operator for environment state |z>.
    In the ortho basis of V (codespace) and V' (lossy codespace):
        [R_z]_{b,i} = <phi_b|R_z|phi'_i>

    The Q (Choi) tensor is:
        Q[b,i,b',i'] = (1/pi) int d^2z [R_z]_{bi} conj([R_z]_{b'i'})
                      = sum_{a,a'} W[a,b,i] env[a,a'] conj(W[a',b',i'])

    where W[a,b,i] = L[a,b] * FM[a,i] with:
        L[a,b] = conj((T_V @ P_ortho)[a,b])  (P_C projected synthesis)
        FM[a,i] = <d'_a|E(P_C)^{-1/2}|phi'_i>  (channel element)

    Args:
        p_mat: (B, B) codespace projector coefficient matrix
        ds_all: (B,) coherent state positions
        gamma: loss probability
        eps: eigenvalue cutoff
        verbose: print diagnostics

    Returns:
        Q: (A_V, A_Vp, A_V, A_Vp) tensor
        subspace_V, subspace_Vp: BosonicSubspace objects
        M_inv_sqrt_ortho: (A_Vp, A_Vp) E(P_C)^{-1/2} in V' ortho basis
    """
    B = ds_all.shape[0]
    t = jnp.sqrt(1.0 - gamma)
    r = jnp.sqrt(gamma)
    ds_prime = t * ds_all

    # Build subspaces
    subspace_V = BosonicSubspace(ds_all, eps=eps)
    subspace_Vp = BosonicSubspace(ds_prime, eps=eps)

    T_V = subspace_V.T       # (B, A_V): synthesis operator for V
    Tp_V = subspace_V.Tp     # (A_V, B): pseudo-inverse for V
    T_Vp = subspace_Vp.T     # (B, A_Vp): synthesis operator for V'
    Tp_Vp = subspace_Vp.Tp   # (A_Vp, B): pseudo-inverse for V'

    A_V = T_V.shape[1]
    A_Vp = T_Vp.shape[1]

    if verbose:
        print(f"  A_V={A_V}, A_Vp={A_Vp}")

    # --- Step A: P_C in ortho basis of V ---
    # P_ortho[b,n] = <phi_b|P_C|phi_n> = (Tp @ p_mat @ Tp^dag)[b,n]
    P_ortho = Tp_V @ p_mat @ dag(Tp_V)  # (A_V, A_V)
    P_ortho = (P_ortho + dag(P_ortho)) / 2.0  # symmetrize

    # --- Step B: E(P_C)^{-1/2} in ortho basis of V' ---
    # E(P_C) coefficient: p'_{ab} = p_{ab} * <r*d_b|r*d_a>
    rd = r * ds_all
    env_overlap = coherent_overlap(rd.reshape(-1, 1), rd.reshape(1, -1))
    # E(P_C) coefficient: p'_{ab} = p_{ab} * <r*d_b|r*d_a> (partial trace formula)
    # env_overlap[a,b] = <rd_a|rd_b>, so we need the conjugate for <rd_b|rd_a>
    p_prime = p_mat * jnp.conj(env_overlap)

    # E(P_C) in ortho basis of V'
    EPC_ortho = Tp_Vp @ p_prime @ dag(Tp_Vp)  # (A_Vp, A_Vp)
    EPC_ortho = (EPC_ortho + dag(EPC_ortho)) / 2.0

    eigvals_E, eigvecs_E = jnp.linalg.eigh(EPC_ortho)
    mask = eigvals_E > eps
    inv_sqrt_E = jnp.where(mask, eigvals_E ** (-0.5), 0.0)
    M_inv_sqrt_ortho = (eigvecs_E * inv_sqrt_E) @ dag(eigvecs_E)  # (A_Vp, A_Vp)

    if verbose:
        n_support = int(jnp.sum(mask))
        print(f"  E(P_C) support dim: {n_support}, top eigvals: {eigvals_E[-3:]}")

    # --- Step C: Build the "channel element" FM ---
    # F[a,k] = <d'_a|phi'_k> = (G' @ T_Vp)[a,k]
    G_prime = coherent_overlap(ds_prime.reshape(-1, 1), ds_prime.reshape(1, -1))
    F = G_prime @ T_Vp  # (B, A_Vp)

    # FM[a,i] = sum_k <d'_a|phi'_k> [E(P_C)^{-1/2}]_{k,i}
    #         = <d'_a| E(P_C)^{-1/2} |phi'_i>
    FM = F @ M_inv_sqrt_ortho  # (B, A_Vp)

    # --- Step D: Build W and compute Q ---
    # For the Kraus operator R_z = P_C E_z^dag E(P_C)^{-1/2}:
    #   [R_z]_{b,i} = sum_n P_ortho[b,n] <phi_n|E_z^dag|psi_i>
    # where |psi_i> = E(P_C)^{-1/2}|phi'_i>.
    #
    # <phi_n|E_z^dag|psi_i> = sum_a conj(T_V[a,n]) <r*d_a|z> FM[a,i]
    #
    # So [R_z]_{b,i} = sum_a L[a,b] <r*d_a|z> FM[a,i]
    # where L[a,b] = sum_n conj(T_V[a,n]) P_ortho[n,b]
    #             = conj((T_V @ P_ortho^T)[a,b])
    # Since P_ortho is Hermitian: L = conj(T_V @ P_ortho)

    L = jnp.conj(T_V @ P_ortho)  # (B, A_V)

    # W[a,b,i] = L[a,b] * FM[a,i]
    W = L[:, :, None] * FM[:, None, :]  # (B, A_V, A_Vp)

    # Q[(b,i),(b',i')] = sum_{a,a'} W[a,b,i] env[a,a'] conj(W[a',b',i'])
    #                   = (W_flat^T @ env @ conj(W_flat))
    W_flat = W.reshape(B, A_V * A_Vp)  # (B, A_V*A_Vp)
    Q_flat = W_flat.T @ env_overlap @ jnp.conj(W_flat)  # (A_V*A_Vp, A_V*A_Vp)

    # Symmetrize (should be Hermitian by construction)
    Q_flat = (Q_flat + dag(Q_flat)) / 2.0

    Q = Q_flat.reshape(A_V, A_Vp, A_V, A_Vp)

    return Q, subspace_V, subspace_Vp, M_inv_sqrt_ortho


# ============================================================
# STEP 3: EIGENDECOMPOSE Q FOR TOP-K KRAUS OPERATORS
# ============================================================

def extract_kraus_from_Q(Q, target_rank=None, eps=1e-8):
    """
    Eigendecompose Q to get the top-K Kraus operators of the transpose channel.

    Q has Hermitian symmetry: Q[b,i,b',i'] = conj(Q[b',i',b,i])
    Reshape to (A_V*A_Vp, A_V*A_Vp) matrix and eigendecompose.

    Args:
        Q: (A_V, A_Vp, A_V, A_Vp) tensor
        target_rank: number of Kraus operators to keep (None = keep all above eps)
        eps: eigenvalue cutoff

    Returns:
        Y_ops: (K, A_V, A_Vp) Kraus operator matrices Y^k_{ai}
        eigenvalues: (K,) kept eigenvalues
        truncation_error: fraction of discarded weight
    """
    A_V, A_Vp = Q.shape[0], Q.shape[1]
    Q_mat = Q.reshape(A_V * A_Vp, A_V * A_Vp)

    # Symmetrize (should be Hermitian by construction)
    Q_mat = (Q_mat + dag(Q_mat)) / 2.0

    eigvals, eigvecs = jnp.linalg.eigh(Q_mat)

    # Sort by descending eigenvalue
    order = jnp.argsort(-eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Keep positive eigenvalues
    total_weight = jnp.sum(jnp.maximum(eigvals, 0.0))

    if target_rank is not None:
        K = target_rank
    else:
        K = int(jnp.sum(eigvals > eps))

    kept_eigvals = eigvals[:K]
    kept_eigvecs = eigvecs[:, :K]  # (A_V*A_Vp, K)

    truncation_error = 1.0 - jnp.sum(jnp.maximum(kept_eigvals, 0.0)) / (total_weight + 1e-30)

    # Build Kraus operators: R_k = sum_{ai} Y^k_{ai} |phi_a><phi'_i|
    # Y^k_{ai} = sqrt(lambda^k) * X^k_{ai}
    Y_ops = jnp.zeros((K, A_V, A_Vp), dtype=jnp.complex64)
    for k in range(K):
        if kept_eigvals[k] > 0:
            Y_k = jnp.sqrt(kept_eigvals[k]) * kept_eigvecs[:, k].reshape(A_V, A_Vp)
        else:
            Y_k = jnp.zeros((A_V, A_Vp), dtype=jnp.complex64)
        Y_ops = Y_ops.at[k].set(Y_k)

    return Y_ops, kept_eigvals, float(truncation_error)


# ============================================================
# STEP 4: FULL PIPELINE
# ============================================================

def build_coherent_transpose_channel(gamma, Delta=0.3, N_trunc=3,
                                      lattice="square", lam=None,
                                      target_rank=None, eps=1e-6, verbose=True):
    """
    Full pipeline: build transpose channel recovery in coherent basis.

    Args:
        gamma: loss probability
        Delta: GKP envelope parameter
        N_trunc: lattice truncation
        lattice: lattice type ("square" or "rect")
        lam: lattice shape parameter (None for default)
        target_rank: Kraus rank to truncate to (None = auto)
        eps: eigenvalue cutoff
        verbose: print diagnostics

    Returns:
        result: dict with all intermediate and final objects
    """
    if verbose:
        print(f"Building coherent-basis transpose channel: "
              f"gamma={gamma}, Delta={Delta}, lattice={lattice}")

    # Step 1: GKP code
    logical_0, logical_1, ds_all, cs_0, cs_1, p_mat = \
        build_gkp_coherent_code(Delta=Delta, N_trunc=N_trunc, lattice=lattice, lam=lam)

    B = ds_all.shape[0]
    if verbose:
        print(f"  Codespace: B={B} coherent states "
              f"(A0={logical_0.cs.shape[0]}, A1={logical_1.cs.shape[0]})")

    # Step 2: Q tensor (ortho-first approach)
    Q, subspace_V, subspace_Vp, M_inv_sqrt = build_Q_tensor(
        p_mat, ds_all, gamma, eps=eps, verbose=verbose
    )

    A_V = Q.shape[0]
    A_Vp = Q.shape[1]
    ds_prime = jnp.sqrt(1.0 - gamma) * ds_all
    if verbose:
        print(f"  Q tensor shape: ({A_V}, {A_Vp}, {A_V}, {A_Vp})")

    # Step 3: Extract Kraus operators
    Y_ops, eigvals, trunc_err = extract_kraus_from_Q(
        Q, target_rank=target_rank, eps=eps
    )

    K = Y_ops.shape[0]
    if verbose:
        print(f"  Kraus rank: {K} (truncation error: {trunc_err:.2e})")
        print(f"  Top eigenvalues: {eigvals[:min(8,K)]}")

    # Step 4: Verify CPTP on support of E(P_C) within V'
    # R†R should equal identity on support of E(P_C), not all of V'
    sum_RdagR = jnp.zeros((A_Vp, A_Vp), dtype=jnp.complex64)
    for k in range(K):
        sum_RdagR = sum_RdagR + dag(Y_ops[k]) @ Y_ops[k]
    # Recompute support projector of E(P_C)
    r = jnp.sqrt(gamma)
    rd = r * ds_all
    env_ov = coherent_overlap(rd.reshape(-1, 1), rd.reshape(1, -1))
    Tp_Vp = subspace_Vp.Tp
    EPC_ortho = Tp_Vp @ (p_mat * jnp.conj(env_ov)) @ dag(Tp_Vp)
    EPC_ortho = (EPC_ortho + dag(EPC_ortho)) / 2.0
    eigvals_E, eigvecs_E = jnp.linalg.eigh(EPC_ortho)
    support_mask = eigvals_E > eps
    n_support = int(jnp.sum(support_mask))
    P_support = eigvecs_E[:, support_mask] @ dag(eigvecs_E[:, support_mask])
    diff_on_support = P_support @ (sum_RdagR - jnp.eye(A_Vp, dtype=jnp.complex64)) @ P_support
    cptp_err = float(jnp.linalg.norm(diff_on_support))
    if verbose:
        print(f"  CPTP error on E(P_C) support (dim={n_support}): {cptp_err:.2e}")

    return {
        'gamma': gamma,
        'Delta': Delta,
        'N_trunc': N_trunc,
        'logical_0': logical_0,
        'logical_1': logical_1,
        'ds_all': ds_all,
        'ds_prime': ds_prime,
        'cs_0': cs_0,
        'cs_1': cs_1,
        'p_mat': p_mat,
        'subspace_V': subspace_V,
        'subspace_Vp': subspace_Vp,
        'M_inv_sqrt_ortho': M_inv_sqrt,
        'Q': Q,
        'Y_ops': Y_ops,
        'eigvals': eigvals,
        'truncation_error': trunc_err,
        'cptp_error': cptp_err,
        'A_V': A_V,
        'A_Vp': A_Vp,
    }


# ============================================================
# VALIDATION: CROSS-CHECK WITH FOCK BASIS
# ============================================================

def validate_against_fock(result, N=GKP_N, loss_rank=10):
    """
    Cross-validate coherent-basis transpose channel against Fock implementation.

    Computes entanglement fidelity of both and compares.
    """
    gamma = result['gamma']
    logical_0 = result['logical_0']
    logical_1 = result['logical_1']
    Y_ops = result['Y_ops']
    subspace_V = result['subspace_V']
    subspace_Vp = result['subspace_Vp']

    K = Y_ops.shape[0]

    # Synthesize coherent-basis Kraus operators to Fock
    # R_k = sum_{ai} Y^k_{ai} |phi_a><phi'_i|
    # |phi_a> in V, |phi'_i> in V'

    # Build Fock basis kets for orthonormal frames
    fock_V = jnp.squeeze(
        jax.vmap(lambda alpha: dqcoherent(N, alpha))(subspace_V.ds)
    )  # (B, N, 1) -> squeeze to (B, N) or handle shape
    fock_Vp = jnp.squeeze(
        jax.vmap(lambda alpha: dqcoherent(N, alpha))(subspace_Vp.ds)
    )

    # Handle potential shape issues
    if fock_V.ndim == 3:
        fock_V = fock_V.squeeze(-1)
    if fock_Vp.ndim == 3:
        fock_Vp = fock_Vp.squeeze(-1)

    # Orthonormal Fock kets: |phi_a> = sum_b T_V[b,a] fock_V[b]
    phi_V = jnp.einsum('ba,bn->an', subspace_V.T, fock_V)   # (A_V, N)
    phi_Vp = jnp.einsum('ba,bn->an', subspace_Vp.T, fock_Vp)  # (A_Vp, N)

    # Build Fock Kraus operators
    recovery_ops_fock = jnp.zeros((K, N, N), dtype=jnp.complex64)
    for k in range(K):
        # R_k = sum_{ai} Y^k[a,i] |phi_a><phi'_i|
        R_k = jnp.einsum('ai,an,im->nm', Y_ops[k], phi_V, jnp.conj(phi_Vp))
        recovery_ops_fock = recovery_ops_fock.at[k].set(R_k)

    # Build Fock logical states
    A0 = logical_0.cs.shape[0]
    fock_V_0 = fock_V[:A0]  # coherent state Fock kets for logical 0
    fock_V_1 = fock_V[A0:A0 + logical_1.cs.shape[0]]  # for logical 1

    psi_0 = jnp.einsum('bn,b->n', fock_V_0, logical_0.cs)
    psi_1 = jnp.einsum('bn,b->n', fock_V_1, logical_1.cs)

    # Normalize
    psi_0 = psi_0 / jnp.sqrt(jnp.real(jnp.dot(jnp.conj(psi_0), psi_0)))
    psi_1 = psi_1 / jnp.sqrt(jnp.real(jnp.dot(jnp.conj(psi_1), psi_1)))
    psi_0 = psi_0.reshape(-1, 1)
    psi_1 = psi_1.reshape(-1, 1)

    # Loss channel
    loss_ops = make_pureloss_fock(gamma, rank=loss_rank, N=N)

    # Entanglement fidelity of coherent-basis recovery (synthesized to Fock)
    psi = [psi_0, psi_1]
    Fe_coherent = 0.0
    for mu in range(2):
        for nu in range(2):
            rho_mn = psi[mu] @ dqdag(psi[nu])
            after_loss = apply_kraus_map_nonorm(loss_ops, rho_mn)
            after_recovery = apply_kraus_map_nonorm(recovery_ops_fock, after_loss)
            Fe_coherent += (dqdag(psi[mu]) @ after_recovery @ psi[nu]).squeeze()
    Fe_coherent = float(jnp.real(Fe_coherent) / 4.0)

    # Reference: Fock-basis transpose channel
    transpose_ops = make_transpose_for_pureloss(loss_ops, logical_0, logical_1)
    Fe_fock_transpose = 0.0
    for mu in range(2):
        for nu in range(2):
            rho_mn = psi[mu] @ dqdag(psi[nu])
            after_loss = apply_kraus_map_nonorm(loss_ops, rho_mn)
            after_recovery = apply_kraus_map_nonorm(transpose_ops, after_loss)
            Fe_fock_transpose += (dqdag(psi[mu]) @ after_recovery @ psi[nu]).squeeze()
    Fe_fock_transpose = float(jnp.real(Fe_fock_transpose) / 4.0)

    # No recovery baseline
    Fe_none = 0.0
    for mu in range(2):
        for nu in range(2):
            rho_mn = psi[mu] @ dqdag(psi[nu])
            after_loss = apply_kraus_map_nonorm(loss_ops, rho_mn)
            Fe_none += (dqdag(psi[mu]) @ after_loss @ psi[nu]).squeeze()
    Fe_none = float(jnp.real(Fe_none) / 4.0)

    print(f"\n  Validation (gamma={gamma}):")
    print(f"    F_e (no recovery):           {Fe_none:.6f}")
    print(f"    F_e (coherent transpose):    {Fe_coherent:.6f}")
    print(f"    F_e (Fock transpose):        {Fe_fock_transpose:.6f}")
    print(f"    Difference:                  {abs(Fe_coherent - Fe_fock_transpose):.2e}")

    return {
        'Fe_coherent_transpose': Fe_coherent,
        'Fe_fock_transpose': Fe_fock_transpose,
        'Fe_none': Fe_none,
        'recovery_ops_fock': recovery_ops_fock,
        'loss_ops': loss_ops,
        'psi_0': psi_0,
        'psi_1': psi_1,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Coherent-Basis Transpose Channel Construction")
    print("=" * 60)

    for gamma in [0.03, 0.05, 0.1]:
        # Use more Kraus operators for larger gamma
        K = 8 if gamma <= 0.05 else 16
        result = build_coherent_transpose_channel(
            gamma=gamma, Delta=0.3, N_trunc=3,
            lattice="square", target_rank=K, verbose=True,
        )
        validation = validate_against_fock(result)
        print()
