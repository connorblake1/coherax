"""
binary_tree_optimizer_claude.py

Binary tree Kraus optimizer for transpose channel recovery on finite GKP codes.

Approach:
1. Compute transpose channel Kraus operators (loss_rank total)
2. Truncate to top target_rank via Choi matrix eigendecomposition
3. Build BinaryKrausTree (depth log2(target_rank), target_rank leaves)
4. For each tree node: optimize T_depth=1 CD+R circuit to match (B_a, B_b)
   using Choi-Hilbert-Schmidt distance as the loss function
5. Reconstruct full Kraus operators by multiplying along tree paths
6. Validate via entanglement fidelity on GKP states

We use direct Frobenius distance (not Choi HS) to pin specific B-node operators,
avoiding left-unitary ambiguity that breaks tree path composition.
Procrustes alignment corrects any residual rotation before path multiplication.
"""

import sys
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from functools import partial
from jaxtyping import Array

from gkp_utils.characteristic_jax_utils import (
    g, channel_from_b,
    make_pureloss_fock, make_transpose_for_pureloss,
    apply_kraus_map_nonorm,
    GKP_N, dqdag, dqtrace, dqcoherent,
)
from gkp_utils.utils import kraus_to_choi, choi_to_kraus, unvec_colwise
from gkp_utils.transpose_channel_claude import (
    build_gkp_states,
    coherent_ket_to_fock,
    entanglement_fidelity,
    entanglement_fidelity_no_recovery,
)


# ============================================================
# VECTORIZED CHANNEL COMPUTATION (avoids slow fori_loop compilation)
# ============================================================

import dynamiqs as dq

# Pre-build annihilation/creation operators for pure JAX displacement
# Fock space dimension for displacement computation.
# Must equal GKP_N to preserve CPTP (unitarity of displacements).
N_OPT = GKP_N
_SQRT_N = jnp.sqrt(jnp.arange(1, N_OPT, dtype=jnp.float32))
_A_OP = jnp.zeros((N_OPT, N_OPT), dtype=jnp.complex64).at[
    jnp.arange(N_OPT - 1), jnp.arange(1, N_OPT)
].set(_SQRT_N.astype(jnp.complex64))
_ADAG_OP = _A_OP.T


def _displace_single(alpha):
    """Pure JAX displacement D(α) = exp(α a† - α* a) in Fock basis."""
    arg = alpha * _ADAG_OP - jnp.conj(alpha) * _A_OP
    return jax.scipy.linalg.expm(arg).astype(jnp.complex64)


def channel_from_b_vec(alphas, betas):
    """
    Vectorized channel computation: K_j = sum_i alpha[j,i] * D(beta[j,i]).

    Uses pure JAX displacement (jax.scipy.linalg.expm) for fast JIT
    compilation and clean gradient tracing.

    Args:
        alphas: (T, N_l) complex coefficients
        betas: (T, N_l) complex displacement amplitudes

    Returns:
        (T, GKP_N, GKP_N) Kraus operators
    """
    def compute_single_kraus(alpha_row, beta_row):
        D_all = jax.vmap(_displace_single)(beta_row)  # (N_l, N, N)
        return jnp.einsum('i,ijk->jk', alpha_row, D_all)

    return jax.vmap(compute_single_kraus)(alphas, betas)


def make_channel_fn(N):
    """Create channel computation function for Fock space of dimension N.

    Returns a function channel_fn(alphas, betas) -> (T, N, N) Kraus operators
    using pure JAX displacement at dimension N. Useful for two-stage optimization
    where Stage 1 uses a smaller Fock space for speed.
    """
    sqrt_n = jnp.sqrt(jnp.arange(1, N, dtype=jnp.float32))
    a_op = jnp.zeros((N, N), dtype=jnp.complex64).at[
        jnp.arange(N - 1), jnp.arange(1, N)
    ].set(sqrt_n.astype(jnp.complex64))
    adag_op = a_op.T

    def displace(alpha):
        arg = alpha * adag_op - jnp.conj(alpha) * a_op
        return jax.scipy.linalg.expm(arg).astype(jnp.complex64)

    def channel_fn(alphas, betas):
        def compute_single_kraus(alpha_row, beta_row):
            D_all = jax.vmap(displace)(beta_row)
            return jnp.einsum('i,ijk->jk', alpha_row, D_all)
        return jax.vmap(compute_single_kraus)(alphas, betas)

    return channel_fn


def _make_fe_loss(channel_fn, loss_ops, psi_0, psi_1, N_depth,
                  beta_penalty=0.0, beta_max=5.0,
                  cptp_penalty=0.0):
    """Build F_e loss function for given channel computation and operators.

    Args:
        beta_penalty: weight for displacement amplitude penalty (0 = no penalty)
        beta_max: max allowed |beta| before penalty kicks in. For N=100,
                  beta_max=5 keeps D(beta) approximately unitary (mean photon < 25).
        cptp_penalty: weight for CPTP violation penalty ||sum K†K - I||² / N.
                      At N_depth=8 with 256 displacement terms, even small
                      non-unitarities of truncated D(β) accumulate. This
                      directly penalizes trace-preservation violation.
    """
    N_l = 2 ** N_depth
    N_fock = psi_0.shape[0]

    def fe_loss(params_flat):
        p = params_flat.reshape(N_depth, 4).astype(jnp.complex64)
        alphas, betas = g(p, N_l)
        recovery_ops = channel_fn(alphas, betas)

        Fe = jnp.float32(0.0)
        psi = [psi_0, psi_1]
        for mu in range(2):
            for nu in range(2):
                rho_mn = psi[mu] @ dqdag(psi[nu])
                after_loss = jnp.sum(
                    jnp.einsum('kij,jl,kml->kim', loss_ops, rho_mn, jnp.conj(loss_ops)),
                    axis=0,
                )
                after_recovery = jnp.sum(
                    jnp.einsum('kij,jl,kml->kim', recovery_ops, after_loss, jnp.conj(recovery_ops)),
                    axis=0,
                )
                Fe = Fe + (dqdag(psi[mu]) @ after_recovery @ psi[nu]).squeeze()

        loss = -jnp.real(Fe) / 4.0

        # Penalty for displacements exceeding Fock space capacity
        # Use mean (not sum) so penalty is independent of N_depth / num Kraus ops
        if beta_penalty > 0:
            excess = jnp.maximum(0.0, jnp.abs(betas)**2 - beta_max**2)
            loss = loss + beta_penalty * jnp.mean(excess)

        # CPTP penalty: ||sum K†K - I||² / N
        # Directly penalizes trace-preservation violation
        if cptp_penalty > 0:
            KdK = jnp.einsum('kji,kjl->il', jnp.conj(recovery_ops), recovery_ops)
            cptp_dev = KdK - jnp.eye(N_fock, dtype=KdK.dtype)
            cptp_loss = jnp.real(jnp.sum(jnp.abs(cptp_dev)**2)) / N_fock
            loss = loss + cptp_penalty * cptp_loss

        return loss

    return fe_loss


# ============================================================
# CHOI TRUNCATION
# ============================================================

def truncate_channel_to_rank(kraus_ops, target_rank=8):
    """
    Truncate a channel to a specified Kraus rank via efficient Gram matrix
    eigendecomposition.

    Instead of forming the full N^2 x N^2 Choi matrix, we work with the
    K x K Gram matrix G = V†V where V = [vec(K_1), ..., vec(K_K)].
    Since rank(Choi) <= K, this is exact and much faster for K << N^2.

    Args:
        kraus_ops: (K, N, N) array of Kraus operators
        target_rank: number of Kraus operators to keep

    Returns:
        truncated_ops: (target_rank, N, N) truncated Kraus operators
        eigenvalues: kept eigenvalues (for diagnostics)
        truncation_error: sum of discarded eigenvalues / total
    """
    K, N, _ = kraus_ops.shape

    # Form V = (N^2, K) matrix of vectorized Kraus ops
    V = kraus_ops.reshape(K, N * N).T  # (N^2, K)

    # Gram matrix G = V†V is only (K, K)
    G = jnp.conj(V.T) @ V  # (K, K)

    # Eigendecompose the small Gram matrix
    eigvals, eigvecs_small = jnp.linalg.eigh(G)  # eigvals (K,), eigvecs (K, K)

    total_weight = jnp.sum(jnp.clip(eigvals, a_min=0.0))
    kept_eigvals = eigvals[-target_rank:]
    kept_eigvecs_small = eigvecs_small[:, -target_rank:]  # (K, target_rank)
    kept_weight = jnp.sum(jnp.clip(kept_eigvals, a_min=0.0))
    truncation_error = float(1.0 - kept_weight / total_weight)

    # Recover Choi eigenvectors: u_i = V @ g_i / sqrt(lambda_i)
    # Then Kraus op = sqrt(lambda_i) * unvec(u_i) = unvec(V @ g_i)
    # So truncated Kraus ops are just linear combinations of original ops
    coeffs = kept_eigvecs_small  # (K, target_rank)

    def build_kraus(i):
        lam = jnp.clip(kept_eigvals[i], a_min=0.0)
        # c = coeffs[:, i] are the combination weights
        c = coeffs[:, i]
        # K_new = sqrt(lam) * unvec(V @ c / sqrt(lam)) = unvec(V @ c)
        # But V @ c = sum_k c_k * vec(K_k), so unvec gives sum_k c_k * K_k
        # We need sqrt(lam) * (normalized eigenvector), which is:
        # sqrt(lam) * V @ c / sqrt(c† G c) = sqrt(lam) * V @ c / sqrt(lam) = V @ c
        # Actually: V @ c has norm sqrt(c† G c) = sqrt(lam), so
        # unvec(V @ c) already has the right normalization
        new_vec = V @ c  # (N^2,)
        return new_vec.reshape(N, N)

    truncated_ops = jax.vmap(build_kraus)(jnp.arange(target_rank))

    return truncated_ops, kept_eigvals, truncation_error


# ============================================================
# PROJECTED BINARY TREE CONSTRUCTION
# ============================================================

def _sqrt_psd(A):
    """Compute matrix square root of a PSD matrix."""
    w, v = jnp.linalg.eigh((A + A.conj().T) * 0.5)
    w = jnp.clip(w, a_min=0.0)
    return (v * jnp.sqrt(w)) @ v.conj().T


class ProjectedBinaryTree:
    """
    Binary tree decomposition in the projected support of sum(K†K).

    Since sum(K†K) for the transpose channel has effective rank ~2
    (the GKP codespace dimension), we:
    1. Project all M_i = K_i†K_i into the r-dimensional support
    2. Normalize to CPTP: K'_i = K_i @ S^{-1/2} where S = sum K_i†K_i
       (so sum K'_i†K'_i = I in the projected space)
    3. Build the tree from the normalized CPTP operators
    4. Store S^{1/2} for reconstructing the actual recovery channel

    The B-node operators are computed as r×r matrices and then embedded
    back into the full N×N space for circuit optimization.
    """

    def __init__(self, kraus_ops, leaf_assign=None, support_thresh=1e-6):
        """
        Args:
            kraus_ops: (K, N, N) Kraus operators
            leaf_assign: permutation of leaf indices (default: identity)
            support_thresh: eigenvalue threshold for support determination
        """
        K_count, N, _ = kraus_ops.shape
        self.K_count = K_count
        self.dim = N
        self.depth = int(np.log2(K_count))
        assert 2**self.depth == K_count, "K_count must be power of 2"

        if leaf_assign is None:
            leaf_assign = jnp.arange(K_count, dtype=int)
        self.leaf_assign = jnp.asarray(leaf_assign, dtype=int)

        # Compute M_i = K_i†K_i
        self.M_orig = jnp.einsum('kij,klj->kil', jnp.conj(kraus_ops), kraus_ops)

        # Find support of sum(M_i) and compute normalization
        Ms_ordered = self.M_orig[self.leaf_assign]
        M_total = Ms_ordered.sum(axis=0)
        eigvals, eigvecs = jnp.linalg.eigh(M_total)
        mask = eigvals > support_thresh * jnp.max(eigvals)
        self.support_rank = int(jnp.sum(mask))
        self.V_r = eigvecs[:, mask]  # (N, r) - basis of support
        self.S_eigvals = eigvals[mask]  # eigenvalues of S in support

        # S^{-1/2} and S^{1/2} in the r-dim support
        self.S_sqrt_proj = jnp.diag(jnp.sqrt(self.S_eigvals))      # (r, r)
        self.S_invsqrt_proj = jnp.diag(1.0 / jnp.sqrt(self.S_eigvals))  # (r, r)

        # S^{1/2} in full space (for reconstruction)
        self.S_sqrt_full = self.V_r @ self.S_sqrt_proj @ self.V_r.conj().T  # (N, N)

        # Project M_i into support: M_i_proj = V_r† M_i V_r
        self.M_proj = jnp.einsum(
            'ir,kij,js->krs', jnp.conj(self.V_r), Ms_ordered, self.V_r
        )  # (K, r, r)

        # Normalize to CPTP: M'_i = S^{-1/2} M_i S^{-1/2}
        # Then sum M'_i = S^{-1/2} S S^{-1/2} = I
        self.M_cptp = jnp.einsum(
            'ra,kab,bs->krs',
            self.S_invsqrt_proj, self.M_proj, self.S_invsqrt_proj
        )  # (K, r, r)

        # Store original Kraus ops for reconstruction
        self.kraus_ops = kraus_ops

        # Build tree from CPTP-normalized operators
        self._build_projected()

    def _build_projected(self):
        """Build tree decomposition from CPTP-normalized operators in r-dim space."""
        r = self.support_rank
        M_leaves = self.M_cptp  # (K, r, r), sum = I

        # Bottom-up: compute M at each level by summing pairs
        Ms = [M_leaves]
        for level in range(self.depth, 0, -1):
            prev = Ms[-1]
            n_nodes = prev.shape[0]
            parent = prev.reshape(n_nodes // 2, 2, r, r).sum(axis=1)
            Ms.append(parent)
        Ms = Ms[::-1]  # Ms[0] = root (should be I), Ms[depth] = leaves

        # Compute sqrt(M) at each level
        self.m_nodes_proj = []
        for d in range(len(Ms)):
            level_sqrtm = []
            for i in range(Ms[d].shape[0]):
                level_sqrtm.append(_sqrt_psd(Ms[d][i]))
            self.m_nodes_proj.append(level_sqrtm)

        # Compute B-nodes: B_a = sqrt(M_child) @ inv(sqrt(M_parent))
        self.B_nodes_proj = []
        self.B_nodes = []
        for d in range(1, len(self.m_nodes_proj)):
            parent = self.m_nodes_proj[d - 1]
            child = self.m_nodes_proj[d]
            level_B_proj = []
            level_B_full = []
            for i in range(len(parent)):
                mp = parent[i]  # (r, r)
                ma = child[2 * i]
                mb = child[2 * i + 1]
                mp_inv = jnp.linalg.pinv(mp)
                ba_proj = ma @ mp_inv
                bb_proj = mb @ mp_inv
                level_B_proj.append((ba_proj, bb_proj))

                # Embed in full space: B_full = V_r @ B_proj @ V_r†
                ba_full = self.V_r @ ba_proj @ self.V_r.conj().T
                bb_full = self.V_r @ bb_proj @ self.V_r.conj().T
                level_B_full.append((ba_full, bb_full))

            self.B_nodes_proj.append(level_B_proj)
            self.B_nodes.append(level_B_full)

    def check(self):
        """Validate tree decomposition (CPTP-normalized, projected space)."""
        r = self.support_rank
        # Reconstruct leaf effects from tree path
        leaf_effects = []
        for leaf_idx in range(self.K_count):
            bits = [(leaf_idx >> k) & 1 for k in range(self.depth - 1, -1, -1)]
            result = jnp.eye(r, dtype=self.M_cptp.dtype)
            parent_in_level = 0
            for d, bit in enumerate(bits):
                ba, bb = self.B_nodes_proj[d][parent_in_level]
                b_op = ba if bit == 0 else bb
                result = b_op @ result
                parent_in_level = (parent_in_level << 1) | bit
            leaf_effects.append(result.conj().T @ result)

        leaf_effects = jnp.stack(leaf_effects)
        # For CPTP tree: K_eff = B_leaf @ ... @ B_root, and since M_root = I,
        # K_eff†K_eff should equal M_leaf (the CPTP-normalized effects)
        diff = float(jnp.linalg.norm(leaf_effects - self.M_cptp))
        # Sum should be I
        comp = float(jnp.linalg.norm(
            leaf_effects.sum(axis=0) - jnp.eye(r, dtype=leaf_effects.dtype)
        ))
        return diff, comp

    @property
    def M_count(self):
        return self.K_count


def build_recovery_tree(truncated_ops, leaf_assign=None):
    """
    Build a ProjectedBinaryTree from truncated Kraus operators.

    Uses projection to the support of sum(K†K) for numerical stability.

    Args:
        truncated_ops: (K, N, N) Kraus operators (K must be power of 2)
        leaf_assign: permutation of leaf indices (default: identity)

    Returns:
        tree: ProjectedBinaryTree object
        node_targets: list of (B_a, B_b) pairs for each internal node,
                      ordered level-by-level (root first), in full N×N space
    """
    K = truncated_ops.shape[0]
    if leaf_assign is None:
        leaf_assign = jnp.arange(K, dtype=int)

    tree = ProjectedBinaryTree(truncated_ops, leaf_assign)

    # Flatten B_nodes into a single list of (B_a, B_b) pairs
    node_targets = []
    for level_pairs in tree.B_nodes:
        for pair in level_pairs:
            node_targets.append(pair)

    return tree, node_targets


# ============================================================
# CHOI-HILBERT-SCHMIDT DISTANCE
# ============================================================

@jax.jit
def choi_hs_distance_sq(ops_a, ops_b):
    """
    Compute ||J_A - J_B||^2_HS using Gram matrix formulation.

    For two channels with Kraus ops {A_k} and {B_l}:
    ||J_A - J_B||^2 = sum|G_AA|^2 + sum|G_BB|^2 - 2*sum|Cross|^2

    where G_AA[k,l] = Tr(A_k† A_l), etc.

    This is unitary-invariant (handles left-unitary freedom in B-nodes).

    Args:
        ops_a: (K, N, N) first set of Kraus operators
        ops_b: (K, N, N) second set of Kraus operators

    Returns:
        scalar: squared Hilbert-Schmidt distance between Choi matrices
    """
    G_AA = jnp.einsum('kij,lij->kl', jnp.conj(ops_a), ops_a)
    G_BB = jnp.einsum('kij,lij->kl', jnp.conj(ops_b), ops_b)
    Cross = jnp.einsum('kij,lij->kl', jnp.conj(ops_a), ops_b)

    dist_sq = (
        jnp.sum(jnp.abs(G_AA)**2)
        + jnp.sum(jnp.abs(G_BB)**2)
        - 2.0 * jnp.sum(jnp.abs(Cross)**2)
    )
    return jnp.real(dist_sq)


@partial(jax.jit, static_argnums=2)
def choi_hs_distance_sq_projected(ops_a, ops_b, r):
    """
    Compute Choi HS distance projected onto an r-dimensional subspace.

    The ops are (K, N, N) but the last r columns/rows of ops_b form the
    support. We project ops_a into this subspace before computing the distance.

    Actually: ops_a and ops_b are both (K, r, r) projected operators.
    This function handles small (r×r) operators efficiently.
    """
    G_AA = jnp.einsum('kij,lij->kl', jnp.conj(ops_a), ops_a)
    G_BB = jnp.einsum('kij,lij->kl', jnp.conj(ops_b), ops_b)
    Cross = jnp.einsum('kij,lij->kl', jnp.conj(ops_a), ops_b)

    dist_sq = (
        jnp.sum(jnp.abs(G_AA)**2)
        + jnp.sum(jnp.abs(G_BB)**2)
        - 2.0 * jnp.sum(jnp.abs(Cross)**2)
    )
    return jnp.real(dist_sq)


# ============================================================
# PER-NODE OPTIMIZER
# ============================================================

def optimize_node(target_Ba, target_Bb, N_depth=4, lr=0.005, steps=3000,
                  restarts=5, random_dist=4.0, random_angle=jnp.pi,
                  verbose=True, V_r=None):
    """
    Optimize a T_depth=1 CD+R circuit to match a single binary tree node.

    The circuit produces 2 Kraus operators via g(params, N_l) -> channel_from_b.
    The target is (B_a, B_b) from the tree decomposition.

    When V_r is provided, the loss is computed in the projected r-dimensional
    subspace for efficiency and to avoid wasting gradient signal on the null space.

    Args:
        target_Ba: (N, N) target B_a operator (full space, embedded)
        target_Bb: (N, N) target B_b operator (full space, embedded)
        N_depth: circuit depth (2^N_depth displacement terms)
        lr: learning rate
        steps: gradient steps per restart
        restarts: random restarts
        random_dist: initialization scale for displacements
        random_angle: initialization scale for angles
        verbose: print progress
        V_r: (N, r) projection matrix for the support subspace (optional)

    Returns:
        best_params: (N_depth, 4) optimized circuit parameters
        best_loss: final projected Choi HS distance squared
    """
    N_l = 2 ** N_depth

    if V_r is not None:
        r = V_r.shape[1]
        # Project targets into support: (2, r, r)
        target_Ba_proj = jnp.conj(V_r.T) @ target_Ba @ V_r
        target_Bb_proj = jnp.conj(V_r.T) @ target_Bb @ V_r
        target_ops_proj = jnp.stack([target_Ba_proj, target_Bb_proj], axis=0)

        def loss_fn(params):
            params_c = params.astype(jnp.complex64)
            alphas, betas = g(params_c, N_l)
            circuit_ops = channel_from_b_vec(alphas, betas)  # (2, N, N)
            # Project circuit ops into support: (2, r, r)
            circuit_proj = jnp.einsum(
                'ir,kij,js->krs', jnp.conj(V_r), circuit_ops, V_r
            )
            # Choi HS distance (unitary-invariant, good convergence)
            # Left-unitary freedom is resolved post-hoc via Procrustes
            return choi_hs_distance_sq_projected(circuit_proj, target_ops_proj, r)
    else:
        target_ops = jnp.stack([target_Ba, target_Bb], axis=0)
        def loss_fn(params):
            params_c = params.astype(jnp.complex64)
            alphas, betas = g(params_c, N_l)
            circuit_ops = channel_from_b(alphas, betas)
            return choi_hs_distance_sq(circuit_ops, target_ops)

    grad_fn = jax.value_and_grad(loss_fn)

    best_loss = float('inf')
    best_params = None

    for restart in range(restarts):
        key = jr.PRNGKey(np.random.randint(100000))
        k1, k2, k3 = jr.split(key, 3)

        params = jnp.zeros((N_depth, 4), jnp.complex64)
        params = params.at[:, 1:3].set(
            2 * random_angle * jr.uniform(key=k2, shape=(N_depth, 2)) - random_angle
        )
        params = params.at[:, 0].set(
            random_dist * jr.normal(key=k1, shape=(N_depth,))
            + random_dist * 1j * jr.normal(key=k3, shape=(N_depth,))
        )

        # Cosine decay schedule
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=lr * 0.1,
            peak_value=lr,
            warmup_steps=steps // 20,
            decay_steps=steps,
            end_value=lr * 0.01,
        )
        optimizer = optax.adam(schedule)
        opt_state = optimizer.init(params)

        last_loss = float('inf')
        for step in range(steps):
            params_c = params.astype(jnp.complex64)
            loss, grads = grad_fn(params_c)
            updates, opt_state = optimizer.update(jnp.conj(grads), opt_state)
            params = optax.apply_updates(params, updates)
            # Zero gamma parameter (rotation convention)
            params = params.at[:, 3].set(jnp.zeros(N_depth))

            if step % 200 == 0:
                current_loss = float(loss)
                if verbose and step % 1000 == 0:
                    print(f"    Restart {restart}, Step {step}: "
                          f"Choi HS^2 = {current_loss:.6e}")
                    sys.stdout.flush()
                if abs(last_loss - current_loss) < 1e-10 and current_loss > 1e-3:
                    if verbose:
                        print(f"    Restart {restart}: early stop at step {step}")
                    break
                last_loss = current_loss

        final_loss = float(loss_fn(params))
        if verbose:
            print(f"    Restart {restart} final: Choi HS^2 = {final_loss:.6e}")
            sys.stdout.flush()

        if final_loss < best_loss:
            best_loss = final_loss
            best_params = params
            if verbose:
                print(f"    >> New best! Choi HS^2 = {best_loss:.6e}")

    return best_params, best_loss


# ============================================================
# FULL TREE OPTIMIZER
# ============================================================

def optimize_full_tree(tree, N_depth=4, lr=0.005, steps=3000, restarts=5,
                       verbose=True, **kwargs):
    """
    Optimize all internal nodes of a ProjectedBinaryTree.

    Iterates level-by-level from root to leaves, optimizing each node
    independently. Uses projected Choi HS loss for efficiency.

    Args:
        tree: ProjectedBinaryTree object (has V_r for projection)
        N_depth: circuit depth per node
        lr: learning rate
        steps: gradient steps per restart per node
        restarts: random restarts per node
        verbose: print progress

    Returns:
        all_params: list of (N_depth, 4) parameter arrays, one per node
        all_losses: list of final projected Choi HS^2 values per node
    """
    all_params = []
    all_losses = []

    V_r = tree.V_r  # projection matrix

    node_idx = 0
    for depth_level, level_pairs in enumerate(tree.B_nodes):
        for node_in_level, (Ba, Bb) in enumerate(level_pairs):
            if verbose:
                print(f"\n  === Node {node_idx} (level {depth_level}, "
                      f"index {node_in_level}) ===")
                sys.stdout.flush()

            params, loss = optimize_node(
                target_Ba=Ba,
                target_Bb=Bb,
                N_depth=N_depth,
                lr=lr,
                steps=steps,
                restarts=restarts,
                verbose=verbose,
                V_r=V_r,
                **kwargs,
            )
            all_params.append(params)
            all_losses.append(loss)
            node_idx += 1

    return all_params, all_losses


# ============================================================
# END-TO-END F_e OPTIMIZER (replaces per-node tree optimization)
# ============================================================

def _build_recovery_ops(params_root, params_left, params_right, N_l):
    """Build 4 recovery Kraus ops from 3 tree circuits (depth-2 binary tree)."""
    alphas_r, betas_r = g(params_root, N_l)
    root_ops = channel_from_b_vec(alphas_r, betas_r)  # (2, N, N)

    alphas_l, betas_l = g(params_left, N_l)
    left_ops = channel_from_b_vec(alphas_l, betas_l)  # (2, N, N)

    alphas_ri, betas_ri = g(params_right, N_l)
    right_ops = channel_from_b_vec(alphas_ri, betas_ri)  # (2, N, N)

    K00 = left_ops[0] @ root_ops[0]
    K01 = left_ops[1] @ root_ops[0]
    K10 = right_ops[0] @ root_ops[1]
    K11 = right_ops[1] @ root_ops[1]
    return jnp.stack([K00, K01, K10, K11])


def optimize_single_circuit(loss_ops, psi_0, psi_1, N_depth=6, lr=0.02,
                            steps=3000, restarts=10, verbose=True,
                            beta_penalty=0.0, beta_max=5.0,
                            cptp_penalty=0.0):
    """
    Optimize a SINGLE CD+R circuit for entanglement fidelity.

    Simpler than the full binary tree: 1 circuit → 2 Kraus operators.
    Closest to SBS architecture. Tests whether CD+R circuits can beat
    the no-recovery baseline at all.

    Returns:
        best_params: (N_depth, 4) circuit parameters
        best_Fe: best F_e achieved
    """
    N_l = 2 ** N_depth
    n_params = N_depth * 4

    fe_loss = _make_fe_loss(channel_from_b_vec, loss_ops, psi_0, psi_1, N_depth,
                            beta_penalty=beta_penalty, beta_max=beta_max,
                            cptp_penalty=cptp_penalty)
    grad_fn = jax.value_and_grad(fe_loss)

    if verbose:
        print("    Compiling single-circuit gradient...")
        sys.stdout.flush()

    best_Fe = -float('inf')
    best_params = None

    for restart in range(restarts):
        key = jr.PRNGKey(np.random.randint(100000))
        k_d, k_a, k_b = jr.split(key, 3)

        params = jnp.zeros((N_depth, 4), dtype=jnp.complex64)
        params = params.at[:, 1:3].set(
            2 * jnp.pi * jr.uniform(key=k_a, shape=(N_depth, 2)) - jnp.pi
        )
        params = params.at[:, 0].set(
            4.0 * jr.normal(key=k_d, shape=(N_depth,))
            + 4.0j * jr.normal(key=k_b, shape=(N_depth,))
        )
        params_flat = params.flatten()

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=lr * 0.1, peak_value=lr,
            warmup_steps=steps // 20, decay_steps=steps,
            end_value=lr * 0.01,
        )
        optimizer = optax.adam(schedule)
        opt_state = optimizer.init(params_flat)

        noise_key = jr.PRNGKey(np.random.randint(100000) + restart)
        last_fe = -1.0
        for step in range(steps):
            neg_fe, grads = grad_fn(params_flat)
            updates, opt_state = optimizer.update(jnp.conj(grads), opt_state)
            params_flat = optax.apply_updates(params_flat, updates)

            # Langevin noise to escape identity basin
            temp_frac = max(0.0, 1.0 - step / (0.6 * steps))
            temperature = 0.05 * temp_frac
            if temperature > 1e-6:
                noise_key, subkey = jr.split(noise_key)
                noise = temperature * jr.normal(subkey, shape=params_flat.shape)
                params_flat = params_flat + noise

            # Zero gamma
            for d in range(N_depth):
                params_flat = params_flat.at[d * 4 + 3].set(0.0)

            if step % 200 == 0:
                current_fe = float(-neg_fe)
                if verbose and step % 500 == 0:
                    print(f"    Restart {restart}, Step {step}: F_e = {current_fe:.6f}")
                    sys.stdout.flush()
                if abs(current_fe - last_fe) < 1e-8 and step > 500:
                    if verbose:
                        print(f"    Restart {restart}: early stop at step {step}")
                    break
                last_fe = current_fe

        final_fe = float(-fe_loss(params_flat))
        if verbose:
            print(f"    Restart {restart} final: F_e = {final_fe:.6f}")
            sys.stdout.flush()

        if final_fe > best_Fe:
            best_Fe = final_fe
            best_params = params_flat.reshape(N_depth, 4)
            if verbose:
                print(f"    >> New best! F_e = {best_Fe:.6f}")

    return best_params, best_Fe


def optimize_single_circuit_twostage(
    loss_ops, psi_0, psi_1, N_depth=8, lr=0.02,
    stage1_steps=2000, stage1_restarts=20, stage1_N=35,
    stage2_steps=1500, stage2_top_k=5,
    beta_penalty=1.0, beta_max=5.0,
    s1_penalty_frac=0.1,
    cptp_penalty=0.0,
    verbose=True,
):
    """Two-stage single-circuit optimizer for higher depths (N_depth >= 8).

    Stage 1: Fast exploration at reduced Fock dimension (N=35).
             ~23x cheaper per expm than N=100, allowing many restarts.
             Uses aggressive Langevin noise for broad exploration.
    Stage 2: Fine-tune top K candidates at full Fock dimension (N=100).
             Lower learning rate, no noise, precise convergence.

    The circuit parameters (alphas, betas from g()) are dimension-independent,
    so parameters found at N=35 transfer directly to N=100.

    Returns:
        best_params: (N_depth, 4) circuit parameters
        best_Fe: best F_e achieved (at full N)
    """
    N_l = 2 ** N_depth
    N_full = psi_0.shape[0]

    # --- Stage 1: Fast exploration at reduced Fock dimension ---
    N1 = stage1_N
    loss_ops_s1 = loss_ops[:, :N1, :N1]
    psi_0_s1 = psi_0[:N1]
    psi_1_s1 = psi_1[:N1]
    psi_0_s1 = psi_0_s1 / jnp.sqrt(jnp.real(dqdag(psi_0_s1) @ psi_0_s1).squeeze())
    psi_1_s1 = psi_1_s1 / jnp.sqrt(jnp.real(dqdag(psi_1_s1) @ psi_1_s1).squeeze())

    channel_fn_s1 = make_channel_fn(N1)
    # Stage 1: configurable penalty fraction (exploration mode)
    beta_penalty_s1 = beta_penalty * s1_penalty_frac
    cptp_penalty_s1 = cptp_penalty * s1_penalty_frac
    fe_loss_s1 = _make_fe_loss(channel_fn_s1, loss_ops_s1, psi_0_s1, psi_1_s1, N_depth,
                               beta_penalty=beta_penalty_s1, beta_max=beta_max,
                               cptp_penalty=cptp_penalty_s1)
    grad_fn_s1 = jax.value_and_grad(fe_loss_s1)

    # Compute baseline at Stage 1 dimension
    fe_none_s1 = 0.0
    psi_s1 = [psi_0_s1, psi_1_s1]
    for mu in range(2):
        rho = psi_s1[mu] @ dqdag(psi_s1[mu])
        after = jnp.sum(
            jnp.einsum('kij,jl,kml->kim', loss_ops_s1, rho, jnp.conj(loss_ops_s1)),
            axis=0,
        )
        fe_none_s1 += float(jnp.real((dqdag(psi_s1[mu]) @ after @ psi_s1[mu]).squeeze()))
    fe_none_s1 /= 2.0

    if verbose:
        print(f"\n  Stage 1: Fast exploration at N={N1}")
        print(f"    F_e baseline (N={N1}): {fe_none_s1:.6f}")
        print(f"    {stage1_restarts} restarts, {stage1_steps} steps each")
        print(f"    beta_penalty: S1={beta_penalty_s1:.4f}, S2={beta_penalty:.4f}, beta_max={beta_max:.2f}")
        if cptp_penalty > 0:
            print(f"    cptp_penalty: S1={cptp_penalty_s1:.4f}, S2={cptp_penalty:.4f}")
        print(f"    Compiling Stage 1 gradient...")
        sys.stdout.flush()

    candidates = []

    init_scale = min(beta_max, 3.0)

    for restart in range(stage1_restarts):
        key = jr.PRNGKey(np.random.randint(100000))
        k_d, k_a, k_b = jr.split(key, 3)

        params = jnp.zeros((N_depth, 4), dtype=jnp.complex64)
        params = params.at[:, 1:3].set(
            2 * jnp.pi * jr.uniform(key=k_a, shape=(N_depth, 2)) - jnp.pi
        )
        params = params.at[:, 0].set(
            init_scale * jr.normal(key=k_d, shape=(N_depth,))
            + init_scale * 1j * jr.normal(key=k_b, shape=(N_depth,))
        )
        params_flat = params.flatten()

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=lr * 0.1, peak_value=lr,
            warmup_steps=stage1_steps // 20, decay_steps=stage1_steps,
            end_value=lr * 0.01,
        )
        optimizer = optax.adam(schedule)
        opt_state = optimizer.init(params_flat)

        noise_key = jr.PRNGKey(np.random.randint(100000) + restart)
        last_fe = -1.0
        for step in range(stage1_steps):
            neg_fe, grads = grad_fn_s1(params_flat)
            updates, opt_state = optimizer.update(jnp.conj(grads), opt_state)
            params_flat = optax.apply_updates(params_flat, updates)

            # Aggressive Langevin noise for broad exploration
            temp_frac = max(0.0, 1.0 - step / (0.7 * stage1_steps))
            temperature = 0.1 * temp_frac
            if temperature > 1e-6:
                noise_key, subkey = jr.split(noise_key)
                noise = temperature * jr.normal(subkey, shape=params_flat.shape)
                params_flat = params_flat + noise

            # Zero gamma
            for d in range(N_depth):
                params_flat = params_flat.at[d * 4 + 3].set(0.0)

            if step % 200 == 0:
                current_fe = float(-neg_fe)
                if verbose and step % 500 == 0:
                    print(f"    S1 R{restart}, Step {step}: F_e = {current_fe:.6f}")
                    sys.stdout.flush()
                if abs(current_fe - last_fe) < 1e-8 and step > 500:
                    break
                last_fe = current_fe

        final_fe = float(-fe_loss_s1(params_flat))
        candidates.append((final_fe, jnp.array(params_flat)))
        if verbose:
            marker = " *" if final_fe > fe_none_s1 else ""
            print(f"    S1 R{restart} final: F_e = {final_fe:.6f}{marker}")
            sys.stdout.flush()

    # Sort candidates by F_e (descending)
    candidates.sort(key=lambda x: x[0], reverse=True)
    n_above = sum(1 for fe, _ in candidates if fe > fe_none_s1)

    if verbose:
        print(f"\n  Stage 1 summary: {n_above}/{stage1_restarts} above baseline")
        print(f"  Top {min(stage2_top_k, len(candidates))} candidates for Stage 2:")
        for i, (fe, _) in enumerate(candidates[:stage2_top_k]):
            print(f"    #{i}: F_e(N={N1}) = {fe:.6f}")
        sys.stdout.flush()

    # --- Stage 2: Fine-tune at full Fock dimension ---
    channel_fn_full = make_channel_fn(N_full)
    fe_loss_full = _make_fe_loss(channel_fn_full, loss_ops, psi_0, psi_1, N_depth,
                                 beta_penalty=beta_penalty, beta_max=beta_max,
                                 cptp_penalty=cptp_penalty)
    grad_fn_full = jax.value_and_grad(fe_loss_full)

    if verbose:
        print(f"\n  Stage 2: Fine-tuning top {stage2_top_k} at N={N_full}")
        print(f"    {stage2_steps} steps each")
        print(f"    Compiling Stage 2 gradient...")
        sys.stdout.flush()

    best_Fe = -float('inf')
    best_params = None

    for i in range(min(stage2_top_k, len(candidates))):
        s1_fe, params_flat = candidates[i]

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=lr * 0.05, peak_value=lr * 0.5,
            warmup_steps=stage2_steps // 20, decay_steps=stage2_steps,
            end_value=lr * 0.005,
        )
        optimizer = optax.adam(schedule)
        opt_state = optimizer.init(params_flat)

        last_fe = -1.0
        for step in range(stage2_steps):
            neg_fe, grads = grad_fn_full(params_flat)
            updates, opt_state = optimizer.update(jnp.conj(grads), opt_state)
            params_flat = optax.apply_updates(params_flat, updates)

            # Zero gamma
            for d in range(N_depth):
                params_flat = params_flat.at[d * 4 + 3].set(0.0)

            if step % 200 == 0:
                current_fe = float(-neg_fe)
                if verbose and step % 500 == 0:
                    print(f"    S2 C{i}, Step {step}: F_e = {current_fe:.6f}")
                    sys.stdout.flush()
                if abs(current_fe - last_fe) < 1e-8 and step > 500:
                    break
                last_fe = current_fe

        final_fe = float(-fe_loss_full(params_flat))
        if verbose:
            print(f"    S2 C{i} final: F_e = {final_fe:.6f}")
            sys.stdout.flush()

        if final_fe > best_Fe:
            best_Fe = final_fe
            best_params = params_flat.reshape(N_depth, 4)
            if verbose:
                print(f"    >> New best! F_e = {best_Fe:.6f}")

    return best_params, best_Fe


def optimize_end_to_end(loss_ops, psi_0, psi_1, N_depth=6, lr=0.005,
                        steps=3000, restarts=10, verbose=True,
                        init_params=None):
    """
    Directly optimize CD+R circuits to maximize entanglement fidelity.

    Uses a depth-2 binary tree structure: 3 circuits (root + 2 children),
    each producing 2 Kraus ops, giving 4 total recovery operators.
    All parameters are optimized jointly to maximize F_e(R∘E).

    Two-phase approach when init_params is provided:
    - Phase 1 (external): per-node Choi HS optimization → good initialization
    - Phase 2 (this function): end-to-end F_e fine-tuning

    Args:
        loss_ops: (K_E, N, N) loss channel Kraus operators
        psi_0, psi_1: (N, 1) normalized Fock kets for |0_L>, |1_L>
        N_depth: circuit depth per node (2^N_depth displacements)
        lr: learning rate
        steps: gradient steps per restart
        restarts: random restarts
        verbose: print progress
        init_params: optional list of 3 (N_depth, 4) arrays for initialization
                     [params_root, params_left, params_right]

    Returns:
        best_params: tuple (params_root, params_left, params_right)
        best_Fe: best entanglement fidelity achieved
    """
    N_l = 2 ** N_depth
    n_params_per_node = N_depth * 4

    def fe_loss(all_params_flat):
        """Negative F_e as loss function."""
        params_root = all_params_flat[:n_params_per_node].reshape(N_depth, 4)
        params_left = all_params_flat[n_params_per_node:2*n_params_per_node].reshape(N_depth, 4)
        params_right = all_params_flat[2*n_params_per_node:].reshape(N_depth, 4)

        recovery_ops = _build_recovery_ops(
            params_root.astype(jnp.complex64),
            params_left.astype(jnp.complex64),
            params_right.astype(jnp.complex64),
            N_l,
        )

        # Compute F_e inline
        Fe = jnp.float32(0.0)
        psi = [psi_0, psi_1]
        for mu in range(2):
            for nu in range(2):
                rho_mn = psi[mu] @ dqdag(psi[nu])
                after_loss = jnp.sum(
                    jnp.einsum('kij,jl,kml->kim', loss_ops, rho_mn, jnp.conj(loss_ops)),
                    axis=0,
                )
                after_recovery = jnp.sum(
                    jnp.einsum('kij,jl,kml->kim', recovery_ops, after_loss, jnp.conj(recovery_ops)),
                    axis=0,
                )
                Fe = Fe + (dqdag(psi[mu]) @ after_recovery @ psi[nu]).squeeze()
        return -jnp.real(Fe) / 4.0

    grad_fn = jax.value_and_grad(fe_loss)

    best_Fe = -float('inf')
    best_params_flat = None

    for restart in range(restarts):
        if restart == 0 and init_params is not None:
            # Use provided initialization (from Phase 1)
            params_flat = jnp.concatenate(
                [p.flatten() for p in init_params]
            ).astype(jnp.complex64)
            if verbose:
                print(f"  [Using Phase 1 initialization for restart 0]")
                sys.stdout.flush()
        else:
            # Random initialization with perturbation from best so far
            key = jr.PRNGKey(np.random.randint(100000))
            k1, k2, k3 = jr.split(key, 3)

            if init_params is not None and restart < restarts // 2:
                # Perturbed initialization (first half of restarts)
                base = jnp.concatenate([p.flatten() for p in init_params])
                noise_scale = 0.5 * (restart + 1)
                noise = noise_scale * jr.normal(k1, shape=base.shape)
                params_flat = (base + noise).astype(jnp.complex64)
            else:
                # Full random initialization (second half)
                params_flat = jnp.zeros(3 * n_params_per_node, dtype=jnp.complex64)
                for node_idx in range(3):
                    k_d, k_a, k_b = jr.split(jr.fold_in(k1, node_idx), 3)
                    offset = node_idx * n_params_per_node
                    node_params = jnp.zeros((N_depth, 4), dtype=jnp.complex64)
                    node_params = node_params.at[:, 1:3].set(
                        2 * jnp.pi * jr.uniform(key=k_a, shape=(N_depth, 2)) - jnp.pi
                    )
                    node_params = node_params.at[:, 0].set(
                        4.0 * jr.normal(key=k_d, shape=(N_depth,))
                        + 4.0j * jr.normal(key=k_b, shape=(N_depth,))
                    )
                    params_flat = params_flat.at[offset:offset + n_params_per_node].set(
                        node_params.flatten()
                    )

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=lr * 0.1,
            peak_value=lr,
            warmup_steps=steps // 20,
            decay_steps=steps,
            end_value=lr * 0.01,
        )
        optimizer = optax.adam(schedule)
        opt_state = optimizer.init(params_flat)

        last_fe = -1.0
        noise_key = jr.PRNGKey(np.random.randint(100000) + restart)
        for step in range(steps):
            neg_fe, grads = grad_fn(params_flat)
            updates, opt_state = optimizer.update(jnp.conj(grads), opt_state)
            params_flat = optax.apply_updates(params_flat, updates)

            # Langevin noise: helps escape identity basin
            # Temperature decays from 0.1 to 0 over first 60% of steps
            temp_frac = max(0.0, 1.0 - step / (0.6 * steps))
            temperature = 0.1 * temp_frac
            if temperature > 1e-6:
                noise_key, subkey = jr.split(noise_key)
                noise = temperature * jr.normal(subkey, shape=params_flat.shape)
                params_flat = params_flat + noise

            # Zero gamma parameters for all 3 nodes
            for node_idx in range(3):
                offset = node_idx * n_params_per_node
                for d in range(N_depth):
                    idx = offset + d * 4 + 3
                    params_flat = params_flat.at[idx].set(0.0)

            if step % 200 == 0:
                current_fe = float(-neg_fe)
                if verbose and step % 500 == 0:
                    print(f"    Restart {restart}, Step {step}: F_e = {current_fe:.6f}")
                    sys.stdout.flush()
                if abs(current_fe - last_fe) < 1e-8 and step > 500:
                    if verbose:
                        print(f"    Restart {restart}: early stop at step {step}")
                    break
                last_fe = current_fe

        final_fe = float(-fe_loss(params_flat))
        if verbose:
            print(f"    Restart {restart} final: F_e = {final_fe:.6f}")
            sys.stdout.flush()

        if final_fe > best_Fe:
            best_Fe = final_fe
            best_params_flat = params_flat
            if verbose:
                print(f"    >> New best! F_e = {best_Fe:.6f}")

    # Unpack best params
    pr = best_params_flat[:n_params_per_node].reshape(N_depth, 4)
    pl = best_params_flat[n_params_per_node:2*n_params_per_node].reshape(N_depth, 4)
    pri = best_params_flat[2*n_params_per_node:].reshape(N_depth, 4)

    return (pr, pl, pri), best_Fe


# ============================================================
# JOINT CHOI HS OPTIMIZER (all 3 circuits jointly)
# ============================================================

def optimize_joint_choi(target_ops, N_depth=6, lr=0.005, steps=3000,
                        restarts=5, verbose=True):
    """
    Jointly optimize 3 tree circuits to minimize Choi HS distance to target.

    Composes 3 CD+R circuits in a depth-2 binary tree to produce 4 recovery
    Kraus operators, then minimizes ||J_recovery - J_target||^2_HS.

    Uses pure JAX displacement (jax.scipy.linalg.expm) for fast JIT
    compilation — traces all 3 circuits in a single gradient computation.

    Key advantages over per-node optimization:
    - No CPTP normalization needed (targets real channel directly)
    - No identity basin (Choi HS has no "do nothing" minimum)
    - Handles left-unitary freedom automatically (Choi HS is invariant)
    - Works with any target rank (recovery has 4 ops, target can have more)

    Args:
        target_ops: (K_T, N, N) target channel Kraus operators
        N_depth: circuit depth per node (2^N_depth displacements)
        lr: learning rate
        steps: gradient steps per restart
        restarts: random restarts
        verbose: print progress

    Returns:
        best_params: tuple (params_root, params_left, params_right)
        best_loss: final Choi HS^2 distance
    """
    N_l = 2 ** N_depth
    n_params_per_node = N_depth * 4

    # Precompute target self-Gram (constant during optimization)
    G_TT = jnp.einsum('kij,lij->kl', jnp.conj(target_ops), target_ops)
    G_TT_norm = jnp.sum(jnp.abs(G_TT)**2)

    def loss_fn(all_params_flat):
        pr = all_params_flat[:n_params_per_node].reshape(N_depth, 4).astype(jnp.complex64)
        pl = all_params_flat[n_params_per_node:2*n_params_per_node].reshape(N_depth, 4).astype(jnp.complex64)
        pri = all_params_flat[2*n_params_per_node:].reshape(N_depth, 4).astype(jnp.complex64)

        recovery_ops = _build_recovery_ops(pr, pl, pri, N_l)

        G_RR = jnp.einsum('kij,lij->kl', jnp.conj(recovery_ops), recovery_ops)
        Cross = jnp.einsum('kij,lij->kl', jnp.conj(recovery_ops), target_ops)

        return jnp.real(
            jnp.sum(jnp.abs(G_RR)**2) + G_TT_norm
            - 2.0 * jnp.sum(jnp.abs(Cross)**2)
        )

    grad_fn = jax.value_and_grad(loss_fn)

    if verbose:
        print("    Compiling gradient (first call)...")
        sys.stdout.flush()

    best_loss = float('inf')
    best_params_flat = None

    for restart in range(restarts):
        key = jr.PRNGKey(np.random.randint(100000))

        params_flat = jnp.zeros(3 * n_params_per_node, dtype=jnp.complex64)
        for node_idx in range(3):
            k_d, k_a, k_b = jr.split(jr.fold_in(key, node_idx), 3)
            offset = node_idx * n_params_per_node
            node_params = jnp.zeros((N_depth, 4), dtype=jnp.complex64)
            node_params = node_params.at[:, 1:3].set(
                2 * jnp.pi * jr.uniform(key=k_a, shape=(N_depth, 2)) - jnp.pi
            )
            node_params = node_params.at[:, 0].set(
                4.0 * jr.normal(key=k_d, shape=(N_depth,))
                + 4.0j * jr.normal(key=k_b, shape=(N_depth,))
            )
            params_flat = params_flat.at[offset:offset + n_params_per_node].set(
                node_params.flatten()
            )

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=lr * 0.1,
            peak_value=lr,
            warmup_steps=steps // 20,
            decay_steps=steps,
            end_value=lr * 0.01,
        )
        optimizer = optax.adam(schedule)
        opt_state = optimizer.init(params_flat)

        last_loss = float('inf')
        for step in range(steps):
            loss, grads = grad_fn(params_flat)
            updates, opt_state = optimizer.update(jnp.conj(grads), opt_state)
            params_flat = optax.apply_updates(params_flat, updates)

            # Zero gamma parameters for all 3 nodes
            for ni in range(3):
                off = ni * n_params_per_node
                for d in range(N_depth):
                    idx = off + d * 4 + 3
                    params_flat = params_flat.at[idx].set(0.0)

            if step % 200 == 0:
                cl = float(loss)
                if verbose and step % 500 == 0:
                    print(f"    Restart {restart}, Step {step}: "
                          f"Choi HS^2 = {cl:.6e}")
                    sys.stdout.flush()
                if abs(last_loss - cl) < 1e-10 and step > 500:
                    if verbose:
                        print(f"    Restart {restart}: early stop at step {step}")
                    break
                last_loss = cl

        final_loss = float(loss_fn(params_flat))
        if verbose:
            print(f"    Restart {restart} final: Choi HS^2 = {final_loss:.6e}")
            sys.stdout.flush()

        if final_loss < best_loss:
            best_loss = final_loss
            best_params_flat = params_flat
            if verbose:
                print(f"    >> New best! Choi HS^2 = {best_loss:.6e}")

    pr = best_params_flat[:n_params_per_node].reshape(N_depth, 4)
    pl = best_params_flat[n_params_per_node:2*n_params_per_node].reshape(N_depth, 4)
    pri = best_params_flat[2*n_params_per_node:].reshape(N_depth, 4)

    return (pr, pl, pri), best_loss


# ============================================================
# KRAUS RECONSTRUCTION FROM OPTIMIZED PARAMS
# ============================================================

def _procrustes_align(circuit_proj, target_proj):
    """
    Find the closest left-unitary U such that U @ circuit ≈ target.

    Solves: min_U ||U @ C[0] - B[0]||^2 + ||U @ C[1] - B[1]||^2
    Solution: U = V @ W† from SVD of X = sum_k B[k] @ C[k]†

    Returns U such that U @ circuit[k] ≈ target[k].
    """
    # X = sum_k target[k] @ circuit[k]†
    X = jnp.einsum('kij,klj->il', target_proj, jnp.conj(circuit_proj))
    U, S, Vt = jnp.linalg.svd(X)
    W = U @ Vt  # closest unitary
    return W


def reconstruct_kraus_from_params(all_params, tree, N_depth, truncated_ops=None,
                                   logical_0=None, logical_1=None):
    """
    Reconstruct full Kraus operators by:
    1. Project circuit ops to r-dim support (avoids null-space contamination)
    2. Procrustes-align each node to its target B-op
    3. Compose aligned r×r operators along root-to-leaf tree paths
    4. Embed using codespace output basis: K = V_code @ W @ K'_proj @ V_r†

    The tree decomposes effects M_i = K†K in V_r (post-loss subspace).
    The recovery operators should map V_r → codespace (not V_r → V_r).
    We compute the global correction W from the original operators' polar
    decomposition, which rotates the tree output into the codespace.

    Args:
        all_params: list of (N_depth, 4) parameter arrays, one per internal node
        tree: ProjectedBinaryTree
        N_depth: circuit depth (to compute N_l)
        truncated_ops: (K, N, N) original truncated Kraus operators (for correction W)
        logical_0, logical_1: CoherentKet GKP states (for codespace basis)

    Returns:
        full_ops: (M, N, N) reconstructed Kraus operators
    """
    N_l = 2 ** N_depth
    depth = tree.depth
    M = tree.M_count
    N = tree.dim
    r = tree.support_rank
    V_r = tree.V_r  # (N, r)

    # Build circuit ops projected to r-dim, with Procrustes alignment
    node_ops_proj = []
    node_idx = 0
    for depth_level, level_pairs in enumerate(tree.B_nodes_proj):
        for node_in_level, (Ba_proj, Bb_proj) in enumerate(level_pairs):
            params = all_params[node_idx]
            params_c = params.astype(jnp.complex64)
            alphas, betas = g(params_c, N_l)
            ops = channel_from_b_vec(alphas, betas)  # (2, N, N)

            # Project to r-dim support
            ops_proj = jnp.einsum(
                'ir,kij,js->krs', jnp.conj(V_r), ops, V_r
            )  # (2, r, r)

            # Procrustes align: find U such that U @ ops_proj ≈ target
            target_proj = jnp.stack([Ba_proj, Bb_proj], axis=0)  # (2, r, r)
            U = _procrustes_align(ops_proj, target_proj)
            ops_aligned = jnp.einsum('ij,kjl->kil', U, ops_proj)  # (2, r, r)

            node_ops_proj.append(ops_aligned)
            node_idx += 1

    # For each leaf, compose projected aligned ops along tree path
    tree_proj_ops = []
    for leaf_idx in range(M):
        bits = [(leaf_idx >> k) & 1 for k in range(depth - 1, -1, -1)]
        result_proj = jnp.eye(r, dtype=jnp.complex64)
        node_offset = 0
        parent_in_level = 0
        for d, bit in enumerate(bits):
            global_node_idx = node_offset + parent_in_level
            ops_d = node_ops_proj[global_node_idx]  # (2, r, r)
            result_proj = ops_d[bit] @ result_proj
            parent_in_level = (parent_in_level << 1) | bit
            if d < depth - 1:
                node_offset += (1 << d)
        tree_proj_ops.append(result_proj)

    tree_proj_ops = jnp.stack(tree_proj_ops)  # (M, r, r)

    # Compute codespace basis and global correction W
    if truncated_ops is not None and logical_0 is not None:
        # Codespace basis: V_code = orthonormalize([psi_0, psi_1])
        psi_0 = coherent_ket_to_fock(logical_0, N)
        psi_1 = coherent_ket_to_fock(logical_1, N)
        psi_0 = psi_0 / jnp.sqrt(jnp.real(dqdag(psi_0) @ psi_0).squeeze())
        psi_1 = psi_1 / jnp.sqrt(jnp.real(dqdag(psi_1) @ psi_1).squeeze())
        # Gram-Schmidt orthogonalize
        overlap = (dqdag(psi_0) @ psi_1).squeeze()
        psi_1_orth = psi_1 - overlap * psi_0
        psi_1_orth = psi_1_orth / jnp.sqrt(
            jnp.real(dqdag(psi_1_orth) @ psi_1_orth).squeeze()
        )
        V_code = jnp.hstack([psi_0, psi_1_orth])  # (N, 2)

        # Compute cross-projections: C_i = V_code† @ K'_actual_i @ V_r
        # K'_actual_i = truncated_ops[leaf_assign[i]] @ S^{-1/2}
        S_invsqrt_full = V_r @ tree.S_invsqrt_proj @ jnp.conj(V_r.T)
        ordered_ops = truncated_ops[tree.leaf_assign]
        K_prime_actual = jnp.einsum(
            'kij,jl->kil', ordered_ops, S_invsqrt_full
        )  # (M, N, N)
        C = jnp.einsum(
            'ir,kij,js->krs', jnp.conj(V_code), K_prime_actual, V_r
        )  # (M, r, r) — cross-projections codespace←V_r

        # Global correction: W = polar(sum_i C_i @ K'_tree_i†)
        X = jnp.einsum('kij,klj->il', C, jnp.conj(tree_proj_ops))
        U_w, S_w, Vt_w = jnp.linalg.svd(X)
        W = U_w @ Vt_w  # (r, r) unitary correction

        # Reconstruct: K_full_i = V_code @ W @ K'_tree_proj_i @ V_r†
        corrected = jnp.einsum('ij,kjl->kil', W, tree_proj_ops)  # (M, r, r)
        full_ops = jnp.einsum(
            'ir,krs,js->kij', V_code, corrected, jnp.conj(V_r)
        )  # (M, N, N)
    else:
        # Fallback: embed as V_r → V_r (less accurate but doesn't need extra info)
        full_ops = jnp.einsum(
            'ir,krs,js->kij', V_r, tree_proj_ops, jnp.conj(V_r)
        )  # (M, N, N)

    return full_ops.astype(jnp.complex64)


# ============================================================
# VALIDATION
# ============================================================

def validate_tree_recovery(reconstructed_ops, loss_ops, logical_0, logical_1,
                           transpose_ops=None, N=GKP_N):
    """
    Validate reconstructed Kraus operators via entanglement fidelity.

    Args:
        reconstructed_ops: (K, N, N) reconstructed recovery Kraus operators
        loss_ops: (K_E, N, N) loss channel Kraus operators
        logical_0, logical_1: CoherentKet GKP logical states
        transpose_ops: (K_T, N, N) transpose channel operators (for comparison)
        N: Fock space truncation

    Returns:
        dict with validation metrics
    """
    psi_0 = coherent_ket_to_fock(logical_0, N)
    psi_1 = coherent_ket_to_fock(logical_1, N)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(dqdag(psi_0) @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(dqdag(psi_1) @ psi_1).squeeze())

    # Entanglement fidelity of tree recovery
    Fe_tree = float(entanglement_fidelity(reconstructed_ops, loss_ops, psi_0, psi_1))

    # No-recovery baseline
    Fe_none = float(entanglement_fidelity_no_recovery(loss_ops, psi_0, psi_1))

    # CPTP check: sum K†K should be approximately I
    kdk = jnp.sum(
        jax.vmap(lambda K: jnp.conj(K.T) @ K)(reconstructed_ops),
        axis=0
    )
    cptp_error = float(jnp.linalg.norm(kdk - jnp.eye(N, dtype=kdk.dtype)))

    result = {
        'Fe_tree': Fe_tree,
        'Fe_none': Fe_none,
        'cptp_error': cptp_error,
    }

    if transpose_ops is not None:
        Fe_transpose = float(entanglement_fidelity(
            transpose_ops, loss_ops, psi_0, psi_1
        ))
        result['Fe_transpose'] = Fe_transpose
        result['gap_to_transpose'] = Fe_transpose - Fe_tree

        # Choi distance between tree and transpose
        choi_dist = float(choi_hs_distance_sq(reconstructed_ops, transpose_ops))
        result['choi_dist_to_transpose'] = choi_dist

    return result


# ============================================================
# END-TO-END DRIVER
# ============================================================

def optimize_tree_recovery(
    gamma=0.05,
    Delta=0.3,
    N_trunc=3,
    target_rank=4,
    N_depth=4,
    lr=0.005,
    steps=3000,
    restarts=5,
    loss_rank=10,
    verbose=True,
    skip_phase1=False,
):
    """
    Binary tree recovery optimization (optionally two-phase).

    Phase 1 (optional): Joint Choi HS — optimize 3 tree circuits to minimize
             Choi HS distance to the full transpose channel.
    Phase 2: End-to-end F_e — directly maximize entanglement fidelity.
             Uses Phase 1 params as initialization if Phase 1 was run.

    Args:
        gamma: loss probability
        Delta: GKP envelope parameter
        N_trunc: coherent state lattice truncation
        target_rank: number of recovery Kraus operators (4 for depth-2 tree)
        N_depth: CD+R circuit depth per tree node
        lr: learning rate
        steps: gradient steps per restart
        restarts: random restarts
        loss_rank: number of loss channel Kraus operators
        verbose: print progress

    Returns:
        dict with all results, operators, and metrics
    """
    t_start = time.time()

    if verbose:
        print(f"{'='*70}")
        mode = "F_e only" if skip_phase1 else "Joint Choi HS + F_e"
        print(f"  Binary Tree Recovery Optimizer ({mode})")
        print(f"  gamma={gamma}, Delta={Delta}, target_rank={target_rank}")
        print(f"  N_depth={N_depth}, lr={lr}, steps={steps}, restarts={restarts}")
        print(f"{'='*70}")
        sys.stdout.flush()

    # Step 1: Build GKP states and channels
    if verbose:
        print("\n[Step 1] Building GKP states and channels...")
        sys.stdout.flush()
    logical_0, logical_1 = build_gkp_states(Delta=Delta, N_trunc=N_trunc)
    loss_ops = make_pureloss_fock(gamma, rank=loss_rank, N=GKP_N)
    transpose_ops = make_transpose_for_pureloss(loss_ops, logical_0, logical_1)

    psi_0 = coherent_ket_to_fock(logical_0, GKP_N)
    psi_1 = coherent_ket_to_fock(logical_1, GKP_N)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(dqdag(psi_0) @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(dqdag(psi_1) @ psi_1).squeeze())

    Fe_none = float(entanglement_fidelity_no_recovery(loss_ops, psi_0, psi_1))
    Fe_transpose = float(entanglement_fidelity(
        transpose_ops, loss_ops, psi_0, psi_1
    ))

    if verbose:
        print(f"  Transpose channel: {transpose_ops.shape[0]} Kraus operators, "
              f"dim={transpose_ops.shape[1]}")
        print(f"  F_e (no recovery):  {Fe_none:.6f}")
        print(f"  F_e (transpose):    {Fe_transpose:.6f}")

    # Diagnostic: truncation ceiling
    truncated_ops, kept_eigvals, trunc_error = truncate_channel_to_rank(
        transpose_ops, target_rank=target_rank
    )
    Fe_trunc = float(entanglement_fidelity(
        truncated_ops, loss_ops, psi_0, psi_1
    ))
    if verbose:
        print(f"  F_e (rank-{target_rank} trunc): {Fe_trunc:.6f} "
              f"(trunc err: {trunc_error:.4e})")

    N_l = 2 ** N_depth
    init_params = None
    Fe_p1 = None
    choi_loss = None

    # Step 2: Phase 1 — Joint Choi HS optimization (optional)
    if not skip_phase1:
        # Target rank-4 truncated transpose (matches our 4 recovery ops)
        phase1_target = truncated_ops
        if verbose:
            print(f"\n[Step 2] Phase 1: Joint Choi HS optimization...")
            print(f"  3 tree circuits → 4 recovery ops, targeting rank-{target_rank} truncated transpose")
            sys.stdout.flush()

        (params_root, params_left, params_right), choi_loss = optimize_joint_choi(
            phase1_target, N_depth=N_depth, lr=lr, steps=steps,
            restarts=restarts, verbose=verbose,
        )

        # Evaluate Phase 1 result
        recovery_ops_p1 = _build_recovery_ops(
            params_root.astype(jnp.complex64),
            params_left.astype(jnp.complex64),
            params_right.astype(jnp.complex64),
            N_l,
        )
        Fe_p1 = float(entanglement_fidelity(recovery_ops_p1, loss_ops, psi_0, psi_1))
        if verbose:
            print(f"\n  Phase 1 result: Choi HS^2 = {choi_loss:.6e}, F_e = {Fe_p1:.6f}")
            sys.stdout.flush()
        init_params = [params_root, params_left, params_right]
    else:
        if verbose:
            print(f"\n[Step 2] Skipping Phase 1 (--skip-phase1)")
            sys.stdout.flush()

    # Step 3: Phase 2 — End-to-end F_e optimization
    fe_lr = lr * 2.0  # Use higher lr for F_e landscape
    if verbose:
        step_label = "Step 3" if not skip_phase1 else "Step 2"
        print(f"\n[{step_label}] Phase 2: End-to-end F_e optimization (lr={fe_lr:.4f})...")
        sys.stdout.flush()

    (params_root, params_left, params_right), best_Fe = optimize_end_to_end(
        loss_ops, psi_0, psi_1,
        N_depth=N_depth, lr=fe_lr, steps=steps, restarts=restarts,
        verbose=verbose,
        init_params=init_params,
    )

    # Build final recovery ops
    reconstructed_ops = _build_recovery_ops(
        params_root.astype(jnp.complex64),
        params_left.astype(jnp.complex64),
        params_right.astype(jnp.complex64),
        N_l,
    )

    # Validate
    val_label = "Step 4" if not skip_phase1 else "Step 3"
    if verbose:
        print(f"\n[{val_label}] Validating...")
        sys.stdout.flush()
    metrics = validate_tree_recovery(
        reconstructed_ops, loss_ops, logical_0, logical_1,
        transpose_ops=transpose_ops, N=GKP_N,
    )

    elapsed = time.time() - t_start

    if verbose:
        print(f"\n{'='*70}")
        print(f"  RESULTS")
        print(f"{'='*70}")
        print(f"  F_e (no recovery):  {metrics['Fe_none']:.6f}")
        if Fe_p1 is not None:
            print(f"  F_e (Phase 1 Choi): {Fe_p1:.6f}")
        print(f"  F_e (Phase 2 F_e):  {metrics['Fe_tree']:.6f}")
        print(f"  F_e (rank-{target_rank} trunc): {Fe_trunc:.6f}")
        print(f"  F_e (transpose):    {metrics.get('Fe_transpose', 'N/A')}")
        print(f"  Gap to transpose:   {metrics.get('gap_to_transpose', 'N/A')}")
        print(f"  CPTP error:         {metrics['cptp_error']:.6e}")
        print(f"  Choi dist to trans: {metrics.get('choi_dist_to_transpose', 'N/A')}")
        print(f"  Truncation error:   {trunc_error:.6e}")
        print(f"  Elapsed: {elapsed:.1f}s")
        sys.stdout.flush()

    return {
        'gamma': gamma,
        'Delta': Delta,
        'target_rank': target_rank,
        'N_depth': N_depth,
        'truncation_error': trunc_error,
        'kept_eigvals': kept_eigvals,
        'Fe_choi_phase': Fe_p1,
        'best_Fe': best_Fe,
        'params_root': params_root,
        'params_left': params_left,
        'params_right': params_right,
        'reconstructed_ops': reconstructed_ops,
        'transpose_ops': transpose_ops,
        'loss_ops': loss_ops,
        'logical_0': logical_0,
        'logical_1': logical_1,
        'metrics': metrics,
        'elapsed_s': elapsed,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Binary tree Kraus optimizer for GKP recovery"
    )
    parser.add_argument("--gamma", type=float, default=0.05,
                        help="Loss parameter")
    parser.add_argument("--Delta", type=float, default=0.3,
                        help="GKP envelope parameter")
    parser.add_argument("--target-rank", type=int, default=4,
                        help="Number of recovery Kraus operators (power of 2)")
    parser.add_argument("--N-depth", type=int, default=4,
                        help="Circuit depth per node")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="Learning rate")
    parser.add_argument("--steps", type=int, default=3000,
                        help="Gradient steps per restart")
    parser.add_argument("--restarts", type=int, default=5,
                        help="Random restarts per node")
    parser.add_argument("--loss-rank", type=int, default=10,
                        help="Loss channel Kraus rank")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with reduced parameters")
    parser.add_argument("--skip-phase1", action="store_true",
                        help="Skip Phase 1 (Choi HS), go directly to F_e optimization")
    parser.add_argument("--single-circuit", action="store_true",
                        help="Optimize a single CD+R circuit (2 Kraus ops) instead of tree")
    parser.add_argument("--two-stage", action="store_true",
                        help="Two-stage optimization: fast exploration at N=35, fine-tune at N=100")
    parser.add_argument("--fast-n", type=int, default=35,
                        help="Fock dimension for Stage 1 of two-stage optimization")
    parser.add_argument("--s1-restarts", type=int, default=20,
                        help="Stage 1 restarts for two-stage optimization")
    parser.add_argument("--s2-top-k", type=int, default=5,
                        help="Number of Stage 1 candidates to fine-tune in Stage 2")
    parser.add_argument("--beta-penalty", type=float, default=1.0,
                        help="Displacement amplitude penalty weight (0=off, uses mean normalization)")
    parser.add_argument("--beta-max", type=float, default=5.0,
                        help="Max |beta| before penalty (should satisfy |beta|^2 << N)")
    parser.add_argument("--s1-penalty-frac", type=float, default=0.1,
                        help="Stage 1 penalty = beta_penalty * s1_penalty_frac (0=no S1 penalty)")
    parser.add_argument("--cptp-penalty", type=float, default=0.0,
                        help="CPTP violation penalty weight for ||sum K†K - I||²/N (0=off)")
    args = parser.parse_args()

    if args.single_circuit:
        # Single CD+R circuit mode
        t_start = time.time()
        from gkp_utils.transpose_channel_claude import (
            build_gkp_states, make_pureloss_fock, make_transpose_for_pureloss,
            entanglement_fidelity, entanglement_fidelity_no_recovery,
            coherent_ket_to_fock,
        )
        logical_0, logical_1 = build_gkp_states(Delta=args.Delta, N_trunc=3)
        loss_ops = make_pureloss_fock(args.gamma, rank=args.loss_rank, N=GKP_N)
        psi_0 = coherent_ket_to_fock(logical_0, GKP_N)
        psi_1 = coherent_ket_to_fock(logical_1, GKP_N)
        psi_0 = psi_0 / jnp.sqrt(jnp.real(dqdag(psi_0) @ psi_0).squeeze())
        psi_1 = psi_1 / jnp.sqrt(jnp.real(dqdag(psi_1) @ psi_1).squeeze())

        Fe_none = float(entanglement_fidelity_no_recovery(loss_ops, psi_0, psi_1))
        N_depth = args.N_depth
        n_steps = 2000 if args.quick else args.steps
        n_restarts = args.restarts

        print(f"{'='*70}")
        mode_str = "Two-Stage" if args.two_stage else "Single"
        print(f"  {mode_str} CD+R Circuit Optimizer")
        print(f"  gamma={args.gamma}, Delta={args.Delta}")
        print(f"  N_depth={N_depth}, lr=0.02, steps={n_steps}, restarts={n_restarts}")
        if args.two_stage:
            print(f"  Stage 1: N={args.fast_n}, {args.s1_restarts} restarts")
            print(f"  Stage 2: N={GKP_N}, top {args.s2_top_k} candidates")
            print(f"  beta_penalty={args.beta_penalty}, beta_max={args.beta_max}")
            if args.cptp_penalty > 0:
                print(f"  cptp_penalty={args.cptp_penalty}")
        print(f"{'='*70}")
        print(f"  F_e (no recovery): {Fe_none:.6f}")
        sys.stdout.flush()

        if args.two_stage:
            best_params, best_Fe = optimize_single_circuit_twostage(
                loss_ops, psi_0, psi_1,
                N_depth=N_depth, lr=0.02,
                stage1_steps=n_steps, stage1_restarts=args.s1_restarts,
                stage1_N=args.fast_n,
                stage2_steps=max(1000, n_steps // 2), stage2_top_k=args.s2_top_k,
                beta_penalty=args.beta_penalty, beta_max=args.beta_max,
                s1_penalty_frac=args.s1_penalty_frac,
                cptp_penalty=args.cptp_penalty,
                verbose=True,
            )
        else:
            best_params, best_Fe = optimize_single_circuit(
                loss_ops, psi_0, psi_1,
                N_depth=N_depth, lr=0.02, steps=n_steps,
                restarts=n_restarts, verbose=True,
                beta_penalty=args.beta_penalty, beta_max=args.beta_max,
                cptp_penalty=args.cptp_penalty,
            )

        # Validate CPTP
        N_l = 2 ** N_depth
        alphas, betas = g(best_params.astype(jnp.complex64), N_l)
        recovery_ops = channel_from_b_vec(alphas, betas)
        # Trace preservation: sum_k K_k† K_k = I
        S = jnp.sum(jax.vmap(lambda K: jnp.conj(K.T) @ K)(recovery_ops), axis=0)
        cptp_err = float(jnp.linalg.norm(S - jnp.eye(GKP_N)))

        # Displacement diagnostics
        max_beta = float(jnp.max(jnp.abs(betas)))
        mean_beta = float(jnp.mean(jnp.abs(betas)))

        elapsed = time.time() - t_start
        print(f"\n{'='*70}")
        print(f"  RESULTS ({mode_str} Circuit)")
        print(f"{'='*70}")
        print(f"  F_e (no recovery): {Fe_none:.6f}")
        print(f"  F_e (best):        {best_Fe:.6f}")
        print(f"  Beat baseline:     {'YES' if best_Fe > Fe_none else 'NO'}")
        print(f"  CPTP error:        {cptp_err:.6e}")
        print(f"  max |beta|:        {max_beta:.3f} (limit: {args.beta_max:.1f})")
        print(f"  mean |beta|:       {mean_beta:.3f}")
        print(f"  Elapsed: {elapsed:.1f}s")
        sys.stdout.flush()

        result = {
            'mode': 'two_stage_circuit' if args.two_stage else 'single_circuit',
            'gamma': args.gamma, 'Delta': args.Delta,
            'N_depth': N_depth, 'best_Fe': best_Fe,
            'Fe_none': Fe_none, 'cptp_error': cptp_err,
            'best_params': best_params, 'elapsed_s': elapsed,
        }
    elif args.quick:
        result = optimize_tree_recovery(
            gamma=args.gamma, Delta=args.Delta,
            target_rank=4, N_depth=args.N_depth,
            lr=0.01, steps=2000, restarts=args.restarts,
            loss_rank=args.loss_rank,
            skip_phase1=args.skip_phase1,
        )
    else:
        result = optimize_tree_recovery(
            gamma=args.gamma, Delta=args.Delta,
            target_rank=args.target_rank, N_depth=args.N_depth,
            lr=args.lr, steps=args.steps, restarts=args.restarts,
            loss_rank=args.loss_rank,
            skip_phase1=args.skip_phase1,
        )

    # Save results
    save_dir = "results_tree_claude"
    os.makedirs(save_dir, exist_ok=True)
    if args.single_circuit:
        tag = "twostage" if args.two_stage else "single"
        save_path = os.path.join(
            save_dir, f"{tag}_gamma{args.gamma:.3f}_depth{args.N_depth}.npy"
        )
    else:
        save_path = os.path.join(
            save_dir, f"tree_gamma{args.gamma:.3f}_rank{result['target_rank']}.npy"
        )
    np.save(save_path, result, allow_pickle=True)
    print(f"\nResults saved to {save_path}")
