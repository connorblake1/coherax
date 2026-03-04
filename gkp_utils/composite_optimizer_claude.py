"""
composite_optimizer_claude.py

Composite pulse-inspired recovery optimization for finite GKP codes.

Key idea from NMR/quantum control: Design R = R_3 o R_2 o R_1 where
systematic errors in R_1 are canceled by R_2 and R_3.

The hypothesis is that composite structure provides robustness to:
  - Over/under-correction
  - Gamma mismatch (optimize at gamma, test at gamma')
  - Multi-round stability

This module provides:
  1. Displacement composition utilities
  2. Composite recovery structures (joint, BB1-style, symmetric)
  3. Optimization strategies for each structure
  4. Robustness analysis tools
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from functools import partial
from jaxtyping import Array
import sys
import time

from gkp_utils.characteristic_jax_utils import (
    CoherentKet,
    gkp_coherent_dm,
    g,
    super_g,
    aOmegab,
    e_n1iaOmegab,
    coherent_overlap,
    dag,
    make_pureloss_fock,
    apply_kraus_map_nonorm,
    channel_from_b,
    GKP_N,
    dqdag,
    dqcoherent,
)


# ============================================================
# DISPLACEMENT COMPOSITION UTILITIES
# ============================================================

@jax.jit
def compose_displacements(alpha_A, beta_A, alpha_B, beta_B):
    """
    Compose two displacement-sum operators K_A and K_B.

    K_A = sum_j alpha_A[j] D(beta_A[j])
    K_B = sum_k alpha_B[k] D(beta_B[k])

    K_A @ K_B = sum_{j,k} alpha_A[j]*alpha_B[k]*exp(-i*Omega(beta_A[j],beta_B[k]))
                          * D(beta_A[j] + beta_B[k])

    where Omega(a,b) = Re(a)*Im(b) - Im(a)*Re(b) is the symplectic form.

    Args:
        alpha_A: (N_A,) complex coefficients for K_A
        beta_A: (N_A,) complex displacements for K_A
        alpha_B: (N_B,) complex coefficients for K_B
        beta_B: (N_B,) complex displacements for K_B

    Returns:
        alpha_AB: (N_A*N_B,) composed coefficients
        beta_AB: (N_A*N_B,) composed displacements
    """
    # Phase from displacement commutation: D(a)D(b) = exp(-i*Omega(a,b)) D(a+b)
    phase = e_n1iaOmegab(beta_A[:, None], beta_B[None, :])  # (N_A, N_B)
    new_alpha = (alpha_A[:, None] * alpha_B[None, :] * phase).reshape(-1)
    new_beta = (beta_A[:, None] + beta_B[None, :]).reshape(-1)
    return new_alpha, new_beta


def compose_kraus_displacements(alpha_A, beta_A, alpha_B, beta_B):
    """
    Compose Kraus operators in displacement representation.

    K_A[mu] @ K_B[nu] for all mu, nu.

    Args:
        alpha_A: (n_A, N_A) coefficients for K_A Kraus operators
        beta_A: (n_A, N_A) displacements for K_A
        alpha_B: (n_B, N_B) coefficients for K_B
        beta_B: (n_B, N_B) displacements for K_B

    Returns:
        alpha_AB: (n_A*n_B, N_A*N_B) composed coefficients
        beta_AB: (n_A*n_B, N_A*N_B) composed displacements
    """
    n_A, N_A = alpha_A.shape
    n_B, N_B = alpha_B.shape
    n_AB = n_A * n_B
    N_AB = N_A * N_B

    # Reshape for broadcasting: (n_A, n_B, N_A, N_B)
    phase = e_n1iaOmegab(
        beta_A[:, None, :, None],  # (n_A, 1, N_A, 1)
        beta_B[None, :, None, :]   # (1, n_B, 1, N_B)
    )

    new_alpha = (
        alpha_A[:, None, :, None] *
        alpha_B[None, :, None, :] *
        phase
    ).reshape(n_AB, N_AB)

    new_beta = (
        beta_A[:, None, :, None] +
        beta_B[None, :, None, :]
    ).reshape(n_AB, N_AB)

    return new_alpha, new_beta


# ============================================================
# COMPOSITE RECOVERY STRUCTURES
# ============================================================

def composite_recovery_joint(params_list, N_l):
    """
    Compose multiple sub-circuits into a single recovery.

    R_total = R_n o ... o R_2 o R_1

    Each R_i is defined by params_list[i] and generates 2 Kraus operators.
    The composed recovery has 2^n Kraus operators.

    Args:
        params_list: list of (N_depth, 4) parameter arrays
        N_l: number of displacement terms per Kraus (2^N_depth)

    Returns:
        alpha_total: (2^n, N_l^n) composed coefficients
        beta_total: (2^n, N_l^n) composed displacements
    """
    # Start with first sub-circuit
    alpha_total, beta_total = g(params_list[0], N_l)

    # Compose remaining sub-circuits (outer applied AFTER inner)
    for params in params_list[1:]:
        alpha_i, beta_i = g(params, N_l)
        alpha_total, beta_total = compose_kraus_displacements(
            alpha_i, beta_i, alpha_total, beta_total
        )

    return alpha_total, beta_total


def bb1_style_recovery(theta_params, phi, N_l, structure="bb1"):
    """
    BB1-style composite recovery with fixed angle relationships.

    BB1 (Broadband 1): R(theta) o R(phi) o R(2*phi) o R(phi) o R(theta)
    where phi = arccos(-theta/(4*pi)) makes the sequence robust to
    pulse area errors.

    For our CD+R circuits, we use a simplified version:
    R_total = R(theta) o R(phi) o R(theta)

    where the relationship between theta and phi is learned.

    Args:
        theta_params: (N_depth, 4) main circuit parameters
        phi: phase rotation between sub-circuits
        N_l: displacement terms per Kraus
        structure: "bb1", "knill", or "corpse"

    Returns:
        alpha_total, beta_total: composed displacement representation
    """
    alpha_theta, beta_theta = g(theta_params, N_l)

    if structure == "bb1":
        # BB1: R(theta) o R(phi) o R(theta)
        # Apply phase rotation to create R(phi) from R(theta)
        phi_params = theta_params.at[:, 1].add(phi)  # Add phase to phi angle
        alpha_phi, beta_phi = g(phi_params, N_l)

        # Compose: R(theta) @ R(phi) @ R(theta)
        alpha_mid, beta_mid = compose_kraus_displacements(
            alpha_phi, beta_phi, alpha_theta, beta_theta
        )
        alpha_total, beta_total = compose_kraus_displacements(
            alpha_theta, beta_theta, alpha_mid, beta_mid
        )

    elif structure == "knill":
        # Knill: R(theta) o R(2*phi) o R(theta) with phi relationship
        phi_params = theta_params.at[:, 1].add(2 * phi)
        alpha_phi, beta_phi = g(phi_params, N_l)

        alpha_mid, beta_mid = compose_kraus_displacements(
            alpha_phi, beta_phi, alpha_theta, beta_theta
        )
        alpha_total, beta_total = compose_kraus_displacements(
            alpha_theta, beta_theta, alpha_mid, beta_mid
        )

    elif structure == "corpse":
        # CORPSE: Uses rotation about different axes
        phi1_params = theta_params.at[:, 1].add(phi)
        phi2_params = theta_params.at[:, 1].add(-phi)

        alpha_phi1, beta_phi1 = g(phi1_params, N_l)
        alpha_phi2, beta_phi2 = g(phi2_params, N_l)

        alpha_mid, beta_mid = compose_kraus_displacements(
            alpha_phi1, beta_phi1, alpha_theta, beta_theta
        )
        alpha_total, beta_total = compose_kraus_displacements(
            alpha_phi2, beta_phi2, alpha_mid, beta_mid
        )
    else:
        raise ValueError(f"Unknown structure: {structure}")

    return alpha_total, beta_total


def symmetric_recovery(params_outer, params_mid, N_l):
    """
    Symmetric composite recovery: R^dag o R_mid o R

    This structure is designed to cancel systematic errors that are
    symmetric about the identity. The outer circuit R and its
    Hermitian conjugate bracket the middle circuit R_mid.

    For displacements: D^dag(beta) = D(-beta), so R^dag flips signs.

    Args:
        params_outer: (N_depth, 4) outer circuit parameters
        params_mid: (N_depth, 4) middle circuit parameters
        N_l: displacement terms per Kraus

    Returns:
        alpha_total, beta_total: composed displacement representation
    """
    alpha_outer, beta_outer = g(params_outer, N_l)
    alpha_mid, beta_mid = g(params_mid, N_l)

    # R^dag: D^dag(beta) = D(-beta), so negate displacements and conjugate coeffs
    alpha_outer_dag = jnp.conj(alpha_outer)
    beta_outer_dag = -beta_outer

    # Compose: R^dag @ R_mid @ R
    alpha_temp, beta_temp = compose_kraus_displacements(
        alpha_mid, beta_mid, alpha_outer, beta_outer
    )
    alpha_total, beta_total = compose_kraus_displacements(
        alpha_outer_dag, beta_outer_dag, alpha_temp, beta_temp
    )

    return alpha_total, beta_total


# ============================================================
# ENTANGLEMENT FIDELITY (COHERENT BASIS)
# ============================================================

@jax.jit
def entanglement_fidelity_displacement(alpha, beta, c_0, d_0, c_1, d_1, gamma):
    """
    Compute entanglement fidelity directly in the coherent basis.

    Fe = (1/4) sum_{mu,nu} sum_k <psi_mu| R_k E(|psi_mu><psi_nu|) R_k^dag |psi_nu>

    where R_k = sum_j alpha[k,j] D(beta[k,j]) and E is pure loss.
    """
    t = jnp.sqrt(1.0 - gamma)
    r = jnp.sqrt(gamma)
    n_kraus = alpha.shape[0]

    cs = [c_0, c_1]
    ds = [d_0, d_1]

    # Precompute environment overlaps
    env = {}
    for mu in range(2):
        env[mu] = {}
        for nu in range(2):
            env[mu][nu] = coherent_overlap(
                r * ds[nu].reshape(-1, 1),
                r * ds[mu].reshape(1, -1),
            )

    Fe = 0.0 + 0j

    for k in range(n_kraus):
        L = {}
        for mu in range(2):
            A_mu = ds[mu].shape[0]
            td_mu = t * ds[mu]

            phase = jnp.exp(-1j * aOmegab(
                beta[k, :, None],
                td_mu[None, :],
            ))

            shifted = beta[k, :, None] + td_mu[None, :]

            ovlp = coherent_overlap(
                ds[mu][:, None, None],
                shifted[None, :, :],
            )

            L[mu] = jnp.einsum(
                'p,j,ja,pja->a',
                jnp.conj(cs[mu]), alpha[k], phase, ovlp,
            )

        for mu in range(2):
            for nu in range(2):
                v_mu = cs[mu] * L[mu]
                v_nu = cs[nu] * L[nu]
                Fe += jnp.conj(v_nu) @ env[mu][nu] @ v_mu

    return jnp.real(Fe) / 4.0


# ============================================================
# OPTIMIZATION STRATEGIES
# ============================================================

def optimize_composite_joint(
    logical_0, logical_1, gamma,
    n_subcircuits=3,
    N_depth=5,
    lr=0.003,
    steps=5000,
    restarts=5,
    random_dist=3.0,
    random_angle=1.0,
    verbose=True,
):
    """
    Option 1: Optimize all sub-circuits jointly.

    Trains R_total = R_n o ... o R_2 o R_1 where each R_i has
    independent parameters.

    Args:
        logical_0, logical_1: CoherentKet GKP logical states
        gamma: loss parameter
        n_subcircuits: number of sub-circuits to compose
        N_depth: CD+R layers per sub-circuit
        lr, steps, restarts: optimization parameters
        random_dist, random_angle: initialization scales
        verbose: print progress

    Returns:
        best_params_list: list of optimized (N_depth, 4) parameters
        best_Fe: best entanglement fidelity
        info: optimization info dict
    """
    N_l = 2 ** N_depth

    def loss_fn(all_params):
        # all_params: (n_subcircuits, N_depth, 4)
        params_list = [all_params[i] for i in range(n_subcircuits)]
        alpha_total, beta_total = composite_recovery_joint(params_list, N_l)
        Fe = entanglement_fidelity_displacement(
            alpha_total, beta_total,
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds,
            gamma,
        )
        return (1.0 - Fe).real

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    eval_fn = jax.jit(loss_fn)

    # Compute baseline (identity)
    Fe_id = float(entanglement_fidelity_displacement(
        jnp.ones((1, 1), dtype=jnp.complex64),
        jnp.zeros((1, 1), dtype=jnp.complex64),
        logical_0.cs, logical_0.ds,
        logical_1.cs, logical_1.ds, gamma
    ))

    if verbose:
        print(f"\n  Composite Joint Optimization")
        print(f"    n_subcircuits={n_subcircuits}, N_depth={N_depth}")
        print(f"    Total Kraus ops: {2**n_subcircuits}")
        print(f"    Displacements per Kraus: {N_l**n_subcircuits}")
        print(f"    Fe_id = {Fe_id:.6f}")
        sys.stdout.flush()

    best_loss = 1.0
    best_params = None
    t_total = time.time()

    for restart in range(restarts):
        key = jr.PRNGKey(np.random.randint(100000))
        k1, k2, k3, key = jr.split(key, 4)

        # Initialize all sub-circuits
        params = jnp.zeros((n_subcircuits, N_depth, 4), jnp.complex64)
        params = params.at[:, :, 1:].set(
            2 * random_angle * jnp.pi * jr.uniform(
                key=k2, shape=(n_subcircuits, N_depth, 3)
            )
        )
        params = params.at[:, :, 0].set(
            random_dist * jr.normal(key=k1, shape=(n_subcircuits, N_depth))
            + random_dist * 1j * jr.normal(key=k3, shape=(n_subcircuits, N_depth))
        )

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=lr * 0.1,
            peak_value=lr,
            warmup_steps=steps // 20,
            decay_steps=steps,
            end_value=lr * 0.01,
        )
        optimizer = optax.adam(schedule)
        opt_state = optimizer.init(params)

        for step in range(steps):
            loss, grads = grad_fn(params)
            updates, opt_state = optimizer.update(jnp.conj(grads), opt_state)
            params = optax.apply_updates(params, updates)
            params = params.at[:, :, 3].set(jnp.zeros((n_subcircuits, N_depth)))

            if verbose and step % 500 == 0:
                Fe = 1.0 - float(loss)
                print(f"    restart {restart}, step {step}: Fe={Fe:.6f}")
                sys.stdout.flush()

        final_loss = float(eval_fn(params))
        Fe_final = 1.0 - final_loss
        if verbose:
            print(f"    restart {restart} final: Fe={Fe_final:.6f}")
            sys.stdout.flush()

        if final_loss < best_loss:
            best_loss = final_loss
            best_params = jnp.array(params)
            if verbose:
                print(f"    >> New best! Fe={1-best_loss:.6f}")
                sys.stdout.flush()

    elapsed = time.time() - t_total
    best_Fe = 1.0 - best_loss
    best_params_list = [best_params[i] for i in range(n_subcircuits)]

    if verbose:
        print(f"\n  Best Fe={best_Fe:.6f}, improvement={best_Fe - Fe_id:+.6f}")
        print(f"  Total time: {elapsed:.1f}s")
        sys.stdout.flush()

    return best_params_list, best_Fe, {
        'Fe_id': Fe_id,
        'n_subcircuits': n_subcircuits,
        'N_depth': N_depth,
        'elapsed': elapsed,
    }


def optimize_bb1_style(
    logical_0, logical_1, gamma,
    N_depth=5,
    structure="bb1",
    lr=0.003,
    steps=5000,
    restarts=5,
    random_dist=3.0,
    random_angle=1.0,
    verbose=True,
):
    """
    Option 2: BB1-style optimization with fixed structure.

    Optimizes theta_params and phi, where the composite structure is:
    R(theta) o R(phi) o R(theta) with specific angle relationships.

    Args:
        logical_0, logical_1: CoherentKet GKP logical states
        gamma: loss parameter
        N_depth: CD+R layers for theta circuit
        structure: "bb1", "knill", or "corpse"
        lr, steps, restarts: optimization parameters
        verbose: print progress

    Returns:
        best_theta_params: optimized (N_depth, 4) parameters
        best_phi: optimized phase
        best_Fe: best entanglement fidelity
        info: optimization info dict
    """
    N_l = 2 ** N_depth

    def loss_fn(packed_params):
        # packed_params: (N_depth+1, 4) where last row encodes phi
        theta_params = packed_params[:-1]
        phi = jnp.real(packed_params[-1, 0])

        alpha_total, beta_total = bb1_style_recovery(
            theta_params, phi, N_l, structure=structure
        )
        Fe = entanglement_fidelity_displacement(
            alpha_total, beta_total,
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds,
            gamma,
        )
        return (1.0 - Fe).real

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    eval_fn = jax.jit(loss_fn)

    # Baseline
    Fe_id = float(entanglement_fidelity_displacement(
        jnp.ones((1, 1), dtype=jnp.complex64),
        jnp.zeros((1, 1), dtype=jnp.complex64),
        logical_0.cs, logical_0.ds,
        logical_1.cs, logical_1.ds, gamma
    ))

    if verbose:
        print(f"\n  BB1-Style Optimization ({structure})")
        print(f"    N_depth={N_depth}, structure={structure}")
        print(f"    Total Kraus ops: 8 (2^3)")
        print(f"    Fe_id = {Fe_id:.6f}")
        sys.stdout.flush()

    best_loss = 1.0
    best_packed = None
    t_total = time.time()

    for restart in range(restarts):
        key = jr.PRNGKey(np.random.randint(100000))
        k1, k2, k3, key = jr.split(key, 4)

        # Initialize: theta_params + phi
        packed = jnp.zeros((N_depth + 1, 4), jnp.complex64)
        packed = packed.at[:-1, 1:].set(
            2 * random_angle * jnp.pi * jr.uniform(key=k2, shape=(N_depth, 3))
        )
        packed = packed.at[:-1, 0].set(
            random_dist * jr.normal(key=k1, shape=(N_depth,))
            + random_dist * 1j * jr.normal(key=k3, shape=(N_depth,))
        )
        # Initialize phi
        packed = packed.at[-1, 0].set(
            2 * jnp.pi * jr.uniform(jr.fold_in(key, 0), ())
        )

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=lr * 0.1,
            peak_value=lr,
            warmup_steps=steps // 20,
            decay_steps=steps,
            end_value=lr * 0.01,
        )
        optimizer = optax.adam(schedule)
        opt_state = optimizer.init(packed)

        for step in range(steps):
            loss, grads = grad_fn(packed)
            updates, opt_state = optimizer.update(jnp.conj(grads), opt_state)
            packed = optax.apply_updates(packed, updates)
            packed = packed.at[:-1, 3].set(jnp.zeros(N_depth))

            if verbose and step % 500 == 0:
                Fe = 1.0 - float(loss)
                print(f"    restart {restart}, step {step}: Fe={Fe:.6f}")
                sys.stdout.flush()

        final_loss = float(eval_fn(packed))
        Fe_final = 1.0 - final_loss
        if verbose:
            print(f"    restart {restart} final: Fe={Fe_final:.6f}")
            sys.stdout.flush()

        if final_loss < best_loss:
            best_loss = final_loss
            best_packed = jnp.array(packed)
            if verbose:
                print(f"    >> New best! Fe={1-best_loss:.6f}")
                sys.stdout.flush()

    elapsed = time.time() - t_total
    best_Fe = 1.0 - best_loss
    best_theta_params = best_packed[:-1]
    best_phi = float(jnp.real(best_packed[-1, 0]))

    if verbose:
        print(f"\n  Best Fe={best_Fe:.6f}, improvement={best_Fe - Fe_id:+.6f}")
        print(f"  Best phi = {best_phi:.4f} rad = {best_phi/jnp.pi:.4f} pi")
        print(f"  Total time: {elapsed:.1f}s")
        sys.stdout.flush()

    return best_theta_params, best_phi, best_Fe, {
        'Fe_id': Fe_id,
        'structure': structure,
        'N_depth': N_depth,
        'elapsed': elapsed,
    }


def optimize_symmetric(
    logical_0, logical_1, gamma,
    N_depth=5,
    lr=0.003,
    steps=5000,
    restarts=5,
    random_dist=3.0,
    random_angle=1.0,
    verbose=True,
):
    """
    Option 3: Symmetric composition R^dag o R_mid o R.

    Optimizes outer and middle circuits independently.

    Args:
        logical_0, logical_1: CoherentKet GKP logical states
        gamma: loss parameter
        N_depth: CD+R layers per sub-circuit
        lr, steps, restarts: optimization parameters
        verbose: print progress

    Returns:
        best_outer: optimized (N_depth, 4) outer parameters
        best_mid: optimized (N_depth, 4) middle parameters
        best_Fe: best entanglement fidelity
        info: optimization info dict
    """
    N_l = 2 ** N_depth

    def loss_fn(packed_params):
        # packed_params: (2, N_depth, 4) for outer and mid
        params_outer = packed_params[0]
        params_mid = packed_params[1]

        alpha_total, beta_total = symmetric_recovery(params_outer, params_mid, N_l)
        Fe = entanglement_fidelity_displacement(
            alpha_total, beta_total,
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds,
            gamma,
        )
        return (1.0 - Fe).real

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    eval_fn = jax.jit(loss_fn)

    # Baseline
    Fe_id = float(entanglement_fidelity_displacement(
        jnp.ones((1, 1), dtype=jnp.complex64),
        jnp.zeros((1, 1), dtype=jnp.complex64),
        logical_0.cs, logical_0.ds,
        logical_1.cs, logical_1.ds, gamma
    ))

    if verbose:
        print(f"\n  Symmetric Optimization (R^dag o R_mid o R)")
        print(f"    N_depth={N_depth}")
        print(f"    Total Kraus ops: 8 (2^3)")
        print(f"    Fe_id = {Fe_id:.6f}")
        sys.stdout.flush()

    best_loss = 1.0
    best_params = None
    t_total = time.time()

    for restart in range(restarts):
        key = jr.PRNGKey(np.random.randint(100000))
        k1, k2, k3, key = jr.split(key, 4)

        params = jnp.zeros((2, N_depth, 4), jnp.complex64)
        params = params.at[:, :, 1:].set(
            2 * random_angle * jnp.pi * jr.uniform(key=k2, shape=(2, N_depth, 3))
        )
        params = params.at[:, :, 0].set(
            random_dist * jr.normal(key=k1, shape=(2, N_depth))
            + random_dist * 1j * jr.normal(key=k3, shape=(2, N_depth))
        )

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=lr * 0.1,
            peak_value=lr,
            warmup_steps=steps // 20,
            decay_steps=steps,
            end_value=lr * 0.01,
        )
        optimizer = optax.adam(schedule)
        opt_state = optimizer.init(params)

        for step in range(steps):
            loss, grads = grad_fn(params)
            updates, opt_state = optimizer.update(jnp.conj(grads), opt_state)
            params = optax.apply_updates(params, updates)
            params = params.at[:, :, 3].set(jnp.zeros((2, N_depth)))

            if verbose and step % 500 == 0:
                Fe = 1.0 - float(loss)
                print(f"    restart {restart}, step {step}: Fe={Fe:.6f}")
                sys.stdout.flush()

        final_loss = float(eval_fn(params))
        Fe_final = 1.0 - final_loss
        if verbose:
            print(f"    restart {restart} final: Fe={Fe_final:.6f}")
            sys.stdout.flush()

        if final_loss < best_loss:
            best_loss = final_loss
            best_params = jnp.array(params)
            if verbose:
                print(f"    >> New best! Fe={1-best_loss:.6f}")
                sys.stdout.flush()

    elapsed = time.time() - t_total
    best_Fe = 1.0 - best_loss
    best_outer = best_params[0]
    best_mid = best_params[1]

    if verbose:
        print(f"\n  Best Fe={best_Fe:.6f}, improvement={best_Fe - Fe_id:+.6f}")
        print(f"  Total time: {elapsed:.1f}s")
        sys.stdout.flush()

    return best_outer, best_mid, best_Fe, {
        'Fe_id': Fe_id,
        'N_depth': N_depth,
        'elapsed': elapsed,
    }


# ============================================================
# ROBUSTNESS ANALYSIS
# ============================================================

def analyze_gamma_robustness(
    alpha, beta,
    logical_0, logical_1,
    gamma_train,
    gamma_range=None,
    n_points=21,
):
    """
    Analyze robustness to gamma mismatch.

    Args:
        alpha, beta: displacement representation of recovery
        logical_0, logical_1: CoherentKet GKP logical states
        gamma_train: gamma used during training
        gamma_range: (gamma_min, gamma_max) or None for default
        n_points: number of test points

    Returns:
        gammas: array of test gamma values
        fidelities: Fe at each gamma
    """
    if gamma_range is None:
        gamma_range = (0.5 * gamma_train, 1.5 * gamma_train)

    gammas = jnp.linspace(gamma_range[0], gamma_range[1], n_points)
    fidelities = []

    for gamma_test in gammas:
        Fe = float(entanglement_fidelity_displacement(
            alpha, beta,
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds,
            float(gamma_test)
        ))
        fidelities.append(Fe)

    return np.array(gammas), np.array(fidelities)


def analyze_multiround_stability(
    alpha, beta,
    logical_0, logical_1,
    gamma,
    n_rounds=10,
):
    """
    Analyze multi-round recovery stability.

    Simulates n_rounds of (loss channel + recovery) and tracks
    the accumulated fidelity.

    This is computed in Fock basis for accuracy.

    Args:
        alpha, beta: displacement representation of recovery
        logical_0, logical_1: CoherentKet GKP logical states
        gamma: loss parameter
        n_rounds: number of loss+recovery rounds

    Returns:
        round_fidelities: Fe after each round
    """
    N = GKP_N
    loss_rank = 10

    # Build Fock operators
    recovery_ops = channel_from_b(alpha, beta)
    loss_ops = make_pureloss_fock(gamma, rank=loss_rank, N=N)

    # Build initial states
    def coherent_to_fock(ck):
        coherents = jnp.squeeze(
            jax.vmap(lambda a: dqcoherent(N, a))(ck.ds)
        )
        psi = jnp.einsum('an,a->n', coherents, ck.cs).reshape(-1, 1)
        psi = psi / jnp.sqrt(jnp.real(dqdag(psi) @ psi).squeeze())
        return psi

    psi_0 = coherent_to_fock(logical_0)
    psi_1 = coherent_to_fock(logical_1)
    psi = [psi_0, psi_1]

    # Initial density matrices
    rho_list = [[psi[mu] @ dqdag(psi[nu]) for nu in range(2)] for mu in range(2)]

    round_fidelities = []

    for _ in range(n_rounds):
        # Apply loss + recovery to each rho[mu][nu]
        for mu in range(2):
            for nu in range(2):
                rho_list[mu][nu] = apply_kraus_map_nonorm(loss_ops, rho_list[mu][nu])
                rho_list[mu][nu] = apply_kraus_map_nonorm(recovery_ops, rho_list[mu][nu])

        # Compute Fe
        Fe = 0.0
        for mu in range(2):
            for nu in range(2):
                Fe += (dqdag(psi[mu]) @ rho_list[mu][nu] @ psi[nu]).squeeze()
        Fe = float(jnp.real(Fe) / 4.0)
        round_fidelities.append(Fe)

    return np.array(round_fidelities)


def compare_single_vs_composite(
    logical_0, logical_1, gamma,
    N_depth=5,
    n_subcircuits=3,
    steps=3000,
    restarts=3,
    verbose=True,
):
    """
    Compare single-circuit vs composite optimization.

    Runs both approaches and compares:
      - Single-round Fe
      - Multi-round Fe stability
      - Robustness to gamma mismatch

    Returns:
        comparison: dict with results for both approaches
    """
    from gkp_utils.coherent_tree_optimizer_claude import (
        entanglement_fidelity_displacement as tree_efe,
        g as tree_g,
    )

    N_l = 2 ** N_depth

    print("=" * 60)
    print("Comparison: Single Circuit vs Composite Recovery")
    print(f"gamma={gamma}, N_depth={N_depth}, n_subcircuits={n_subcircuits}")
    print("=" * 60)

    # --- Single Circuit Optimization ---
    print("\n[1] Single Circuit (baseline)")

    def single_loss_fn(params):
        alpha, beta = g(params, N_l)
        Fe = entanglement_fidelity_displacement(
            alpha, beta,
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds, gamma
        )
        return (1.0 - Fe).real

    single_grad_fn = jax.jit(jax.value_and_grad(single_loss_fn))

    best_single_loss = 1.0
    best_single_params = None

    for restart in range(restarts):
        key = jr.PRNGKey(np.random.randint(100000))
        k1, k2, k3, key = jr.split(key, 4)

        params = jnp.zeros((N_depth, 4), jnp.complex64)
        params = params.at[:, 1:].set(
            2 * jnp.pi * jr.uniform(key=k2, shape=(N_depth, 3))
        )
        params = params.at[:, 0].set(
            3.0 * jr.normal(key=k1, shape=(N_depth,))
            + 3.0 * 1j * jr.normal(key=k3, shape=(N_depth,))
        )

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.003 * 0.1, peak_value=0.003,
            warmup_steps=steps // 20, decay_steps=steps,
            end_value=0.003 * 0.01,
        )
        optimizer = optax.adam(schedule)
        opt_state = optimizer.init(params)

        for step in range(steps):
            loss, grads = single_grad_fn(params)
            updates, opt_state = optimizer.update(jnp.conj(grads), opt_state)
            params = optax.apply_updates(params, updates)
            params = params.at[:, 3].set(jnp.zeros(N_depth))

            if verbose and step % 500 == 0:
                print(f"    restart {restart}, step {step}: Fe={1-float(loss):.6f}")
                sys.stdout.flush()

        final_loss = float(single_loss_fn(params))
        if final_loss < best_single_loss:
            best_single_loss = final_loss
            best_single_params = jnp.array(params)

    single_Fe = 1.0 - best_single_loss
    alpha_single, beta_single = g(best_single_params, N_l)
    print(f"  Single circuit Fe: {single_Fe:.6f}")

    # --- Composite Optimization ---
    print(f"\n[2] Composite ({n_subcircuits} sub-circuits)")

    composite_params_list, composite_Fe, _ = optimize_composite_joint(
        logical_0, logical_1, gamma,
        n_subcircuits=n_subcircuits,
        N_depth=N_depth,
        steps=steps,
        restarts=restarts,
        verbose=verbose,
    )
    alpha_comp, beta_comp = composite_recovery_joint(composite_params_list, N_l)
    print(f"  Composite Fe: {composite_Fe:.6f}")

    # --- Gamma Robustness Analysis ---
    print("\n[3] Gamma Robustness Analysis")

    gammas_single, fes_single = analyze_gamma_robustness(
        alpha_single, beta_single, logical_0, logical_1, gamma
    )
    gammas_comp, fes_comp = analyze_gamma_robustness(
        alpha_comp, beta_comp, logical_0, logical_1, gamma
    )

    single_robustness = np.std(fes_single)
    comp_robustness = np.std(fes_comp)

    print(f"  Single: std(Fe) = {single_robustness:.6f}")
    print(f"  Composite: std(Fe) = {comp_robustness:.6f}")

    # --- Multi-Round Stability ---
    print("\n[4] Multi-Round Stability (10 rounds)")

    single_rounds = analyze_multiround_stability(
        alpha_single, beta_single, logical_0, logical_1, gamma, n_rounds=10
    )
    comp_rounds = analyze_multiround_stability(
        alpha_comp, beta_comp, logical_0, logical_1, gamma, n_rounds=10
    )

    print(f"  Single: Fe after 10 rounds = {single_rounds[-1]:.6f}")
    print(f"  Composite: Fe after 10 rounds = {comp_rounds[-1]:.6f}")

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Single-round:     Single={single_Fe:.6f}, Composite={composite_Fe:.6f}")
    print(f"  Gamma robustness: Single={single_robustness:.6f}, Composite={comp_robustness:.6f}")
    print(f"  Multi-round (10): Single={single_rounds[-1]:.6f}, Composite={comp_rounds[-1]:.6f}")
    print("=" * 60)

    return {
        'single': {
            'params': best_single_params,
            'Fe': single_Fe,
            'alpha': alpha_single,
            'beta': beta_single,
            'gamma_robustness': (gammas_single, fes_single),
            'multiround': single_rounds,
        },
        'composite': {
            'params_list': composite_params_list,
            'Fe': composite_Fe,
            'alpha': alpha_comp,
            'beta': beta_comp,
            'gamma_robustness': (gammas_comp, fes_comp),
            'multiround': comp_rounds,
        },
    }


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Composite Pulse-Inspired Recovery Optimization")
    print("=" * 60)

    # Parameters
    gamma = 0.1
    Delta = 0.3
    N_trunc = 3
    N_depth = 5

    # Build GKP states
    logical_0 = gkp_coherent_dm(mu=0, N_trunc=N_trunc, Delta=Delta, lattice='square')
    logical_1 = gkp_coherent_dm(mu=1, N_trunc=N_trunc, Delta=Delta, lattice='square')

    # Compute baseline (identity recovery)
    Fe_id = float(entanglement_fidelity_displacement(
        jnp.ones((1, 1), dtype=jnp.complex64),
        jnp.zeros((1, 1), dtype=jnp.complex64),
        logical_0.cs, logical_0.ds,
        logical_1.cs, logical_1.ds, gamma
    ))
    print(f"\ngamma={gamma}, Delta={Delta}, Fe_id={Fe_id:.6f}")

    # --- Test all three composite strategies ---

    # 1. Joint optimization (3 sub-circuits)
    print("\n" + "=" * 60)
    print("Strategy 1: Joint Optimization (3 sub-circuits)")
    print("=" * 60)

    params_list_joint, Fe_joint, info_joint = optimize_composite_joint(
        logical_0, logical_1, gamma,
        n_subcircuits=3,
        N_depth=N_depth,
        steps=2000,
        restarts=3,
        verbose=True,
    )

    # 2. BB1-style
    print("\n" + "=" * 60)
    print("Strategy 2: BB1-Style Composite")
    print("=" * 60)

    theta_params_bb1, phi_bb1, Fe_bb1, info_bb1 = optimize_bb1_style(
        logical_0, logical_1, gamma,
        N_depth=N_depth,
        structure="bb1",
        steps=2000,
        restarts=3,
        verbose=True,
    )

    # 3. Symmetric
    print("\n" + "=" * 60)
    print("Strategy 3: Symmetric (R^dag o R_mid o R)")
    print("=" * 60)

    outer_params, mid_params, Fe_sym, info_sym = optimize_symmetric(
        logical_0, logical_1, gamma,
        N_depth=N_depth,
        steps=2000,
        restarts=3,
        verbose=True,
    )

    # --- Final comparison ---
    print("\n" + "=" * 60)
    print("Final Comparison")
    print("=" * 60)
    print(f"  Identity recovery:   Fe = {Fe_id:.6f}")
    print(f"  Joint composite:     Fe = {Fe_joint:.6f} (+{Fe_joint - Fe_id:+.6f})")
    print(f"  BB1-style composite: Fe = {Fe_bb1:.6f} (+{Fe_bb1 - Fe_id:+.6f})")
    print(f"  Symmetric composite: Fe = {Fe_sym:.6f} (+{Fe_sym - Fe_id:+.6f})")

    # --- Robustness comparison for best method ---
    best_method = max([
        ('joint', Fe_joint),
        ('bb1', Fe_bb1),
        ('symmetric', Fe_sym)
    ], key=lambda x: x[1])

    print(f"\nBest method: {best_method[0]} with Fe={best_method[1]:.6f}")

    # Run full comparison against single circuit
    print("\n" + "=" * 60)
    print("Full Comparison: Single vs Composite")
    print("=" * 60)

    comparison = compare_single_vs_composite(
        logical_0, logical_1, gamma,
        N_depth=N_depth,
        n_subcircuits=3,
        steps=2000,
        restarts=2,
        verbose=True,
    )
