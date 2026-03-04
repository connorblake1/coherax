"""
lyapunov_optimizer_claude.py

Lyapunov-style stability-constrained optimization for CD+R recovery circuits.

Key insight: Standard single-round optimization can find circuits with high Fe_1
but poor Fe_n for n > 1 due to accumulated damage. This module enforces that
Fe is non-decreasing (or slowly decreasing) across rounds to prevent accumulated
damage via Lyapunov-style stability constraints.

Constraint formulations:
  1. Penalty method: loss = (1 - Fe_1) + lambda * max(0, Fe_1 - Fe_2/alpha)
  2. Ratio constraint: Fe_2/Fe_1 >= alpha
  3. Multi-step: check Fe_1, Fe_2, ..., Fe_k

Uses CMA-ES for optimization since the landscape is highly multimodal.

Usage:
    python -m gkp_utils.lyapunov_optimizer_claude
    python -m gkp_utils.lyapunov_optimizer_claude --gamma 0.05 --alpha 0.95
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import time
import sys
import os
import argparse
from functools import partial

from gkp_utils.characteristic_jax_utils import (
    CoherentKet,
    gkp_coherent_dm,
    g,
    channel_from_b,
    make_pureloss_fock,
    apply_kraus_map_nonorm,
    dag,
    dqdag,
    GKP_N,
    dqcoherent,
    coherent_overlap,
    aOmegab,
)


# ============================================================
# GKP STATE SETUP
# ============================================================

def build_gkp_states(Delta=0.3, N_trunc=3, lattice="square"):
    """Build GKP logical states in coherent basis."""
    logical_0 = gkp_coherent_dm(mu=0, N_trunc=N_trunc, Delta=Delta, lattice=lattice)
    logical_1 = gkp_coherent_dm(mu=1, N_trunc=N_trunc, Delta=Delta, lattice=lattice)
    return logical_0, logical_1


def coherent_ket_to_fock(ck, N=GKP_N):
    """Convert CoherentKet to Fock-basis ket."""
    coherents = jax.vmap(lambda alpha: dqcoherent(N, alpha))(ck.ds)
    psi = jnp.einsum('ijk,i->jk', coherents, ck.cs)
    return psi / jnp.sqrt(jnp.real(jnp.conj(psi).T @ psi).squeeze())


# ============================================================
# ENTANGLEMENT FIDELITY (COHERENT BASIS)
# ============================================================

def entanglement_fidelity_displacement(alpha, beta, c_0, d_0, c_1, d_1, gamma):
    """
    Compute entanglement fidelity directly in the coherent basis.

    Fe = (1/4) sum_{mu,nu} sum_k <psi_mu| R_k E(|psi_mu><psi_nu|) R_k^dag |psi_nu>

    where R_k = sum_j alpha[k,j] D(beta[k,j]) are recovery operators and
    E is the pure loss channel with parameter gamma.
    """
    t = jnp.sqrt(1.0 - gamma)
    r = jnp.sqrt(gamma)
    n_kraus = alpha.shape[0]

    cs = [c_0, c_1]
    ds = [d_0, d_1]

    # Precompute env overlaps for all (mu,nu) pairs
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
# MULTI-ROUND FIDELITY COMPUTATION
# ============================================================

def compute_fe_round(params, logical_0, logical_1, gamma, N_depth, n_rounds=1, N=GKP_N):
    """
    Compute entanglement fidelity after n_rounds of (loss -> recovery).

    This uses Fock-basis simulation for accuracy.
    """
    N_l = 2 ** N_depth
    alpha, beta = g(params, N_l)
    recovery_ops = channel_from_b(alpha, beta)
    loss_ops = make_pureloss_fock(gamma, rank=10, N=N)

    psi_0 = coherent_ket_to_fock(logical_0, N)
    psi_1 = coherent_ket_to_fock(logical_1, N)
    psi = [psi_0, psi_1]

    # Initialize rho_{mu,nu} = |mu><nu|
    rhos = {}
    for mu in range(2):
        for nu in range(2):
            rhos[(mu, nu)] = psi[mu] @ dqdag(psi[nu])

    # Apply n_rounds of (loss -> recovery)
    for _ in range(n_rounds):
        for mu in range(2):
            for nu in range(2):
                rho_after_loss = apply_kraus_map_nonorm(loss_ops, rhos[(mu, nu)])
                rhos[(mu, nu)] = apply_kraus_map_nonorm(recovery_ops, rho_after_loss)

    # Compute Fe
    Fe = 0.0 + 0j
    for mu in range(2):
        for nu in range(2):
            Fe += (dqdag(psi[mu]) @ rhos[(mu, nu)] @ psi[nu]).squeeze()
    return float(jnp.real(Fe) / 4.0)


def compute_fe_trajectory(params, logical_0, logical_1, gamma, N_depth, n_rounds=20, N=GKP_N):
    """
    Compute Fe trajectory over n_rounds of stabilization.

    Returns list of Fe values: [Fe_1, Fe_2, ..., Fe_n_rounds]
    """
    N_l = 2 ** N_depth
    alpha, beta = g(params, N_l)
    recovery_ops = channel_from_b(alpha, beta)
    loss_ops = make_pureloss_fock(gamma, rank=10, N=N)

    psi_0 = coherent_ket_to_fock(logical_0, N)
    psi_1 = coherent_ket_to_fock(logical_1, N)
    psi = [psi_0, psi_1]

    rhos = {}
    for mu in range(2):
        for nu in range(2):
            rhos[(mu, nu)] = psi[mu] @ dqdag(psi[nu])

    fe_history = []
    for _ in range(n_rounds):
        for mu in range(2):
            for nu in range(2):
                rho_after_loss = apply_kraus_map_nonorm(loss_ops, rhos[(mu, nu)])
                rhos[(mu, nu)] = apply_kraus_map_nonorm(recovery_ops, rho_after_loss)

        Fe = 0.0 + 0j
        for mu in range(2):
            for nu in range(2):
                Fe += (dqdag(psi[mu]) @ rhos[(mu, nu)] @ psi[nu]).squeeze()
        fe_history.append(float(jnp.real(Fe) / 4.0))

    return fe_history


# ============================================================
# LYAPUNOV-CONSTRAINED LOSS FUNCTIONS
# ============================================================

def make_stability_loss_penalty(
    logical_0, logical_1, gamma, N_depth,
    alpha_stability=0.95,
    lambda_penalty=10.0,
    n_check_rounds=2,
):
    """
    Build loss function with stability penalty.

    Loss = (1 - Fe_1) + lambda * max(0, Fe_1 - Fe_2/alpha)

    This penalizes solutions where Fe_2 < alpha * Fe_1.
    """
    N_l = 2 ** N_depth

    def loss_fn(params):
        alpha, beta = g(params, N_l)

        # Compute Fe_1 analytically (fast)
        Fe_1 = entanglement_fidelity_displacement(
            alpha, beta,
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds,
            gamma,
        )

        # Compute Fe_2 in Fock basis (accurate)
        Fe_2 = compute_fe_round(params, logical_0, logical_1, gamma, N_depth, n_rounds=2)

        # Primary objective: maximize Fe_1
        primary_loss = 1.0 - Fe_1

        # Stability constraint: Fe_2 >= alpha * Fe_1
        # Penalty = max(0, Fe_1 - Fe_2/alpha) = max(0, (alpha*Fe_1 - Fe_2)/alpha)
        stability_violation = jnp.maximum(0.0, alpha_stability * Fe_1 - Fe_2)
        penalty = lambda_penalty * stability_violation

        return float(primary_loss + penalty)

    return loss_fn


def make_stability_loss_ratio(
    logical_0, logical_1, gamma, N_depth,
    alpha_stability=0.95,
    lambda_penalty=10.0,
):
    """
    Build loss function with ratio constraint.

    Directly penalizes when Fe_2/Fe_1 < alpha.
    """
    N_l = 2 ** N_depth

    def loss_fn(params):
        alpha, beta = g(params, N_l)

        Fe_1 = entanglement_fidelity_displacement(
            alpha, beta,
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds,
            gamma,
        )

        Fe_2 = compute_fe_round(params, logical_0, logical_1, gamma, N_depth, n_rounds=2)

        primary_loss = 1.0 - Fe_1

        # Ratio constraint
        ratio = Fe_2 / jnp.maximum(Fe_1, 1e-6)
        ratio_violation = jnp.maximum(0.0, alpha_stability - ratio)
        penalty = lambda_penalty * ratio_violation

        return float(primary_loss + penalty)

    return loss_fn


def make_stability_loss_multistep(
    logical_0, logical_1, gamma, N_depth,
    alpha_stability=0.95,
    lambda_penalty=5.0,
    check_rounds=[1, 2, 3, 5],
):
    """
    Build loss function with multi-step stability constraint.

    Checks Fe at multiple rounds and penalizes violations at each step.
    """
    N_l = 2 ** N_depth

    def loss_fn(params):
        alpha, beta = g(params, N_l)

        # Compute Fe_1 analytically
        Fe_1 = entanglement_fidelity_displacement(
            alpha, beta,
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds,
            gamma,
        )

        primary_loss = 1.0 - Fe_1

        # Get Fe trajectory for check_rounds
        max_round = max(check_rounds)
        fe_trajectory = compute_fe_trajectory(
            params, logical_0, logical_1, gamma, N_depth, n_rounds=max_round
        )

        # Compute cumulative penalty
        total_penalty = 0.0
        prev_fe = Fe_1
        for r in check_rounds:
            fe_r = fe_trajectory[r - 1]  # 0-indexed
            # Penalty if Fe_r < alpha * Fe_{r-1}
            # We use alpha^r to allow gradual decay
            expected_fe = (alpha_stability ** r) * Fe_1
            violation = jnp.maximum(0.0, expected_fe - fe_r)
            total_penalty += violation
            prev_fe = fe_r

        penalty = lambda_penalty * total_penalty

        return float(primary_loss + penalty)

    return loss_fn


# ============================================================
# CMA-ES OPTIMIZATION
# ============================================================

def optimize_cmaes_lyapunov(
    logical_0, logical_1, gamma,
    N_depth=6,
    constraint_type="penalty",  # "penalty", "ratio", "multistep"
    alpha_stability=0.95,
    lambda_penalty=10.0,
    check_rounds=[1, 2, 3, 5],
    popsize=80,
    maxiter=2000,
    sigma0=3.0,
    seed=42,
    verbose=True,
):
    """
    CMA-ES optimization with Lyapunov stability constraint.

    Args:
        logical_0, logical_1: CoherentKet GKP logical states
        gamma: loss parameter
        N_depth: CD+R circuit depth (N_l = 2^N_depth)
        constraint_type: "penalty", "ratio", or "multistep"
        alpha_stability: stability coefficient (Fe_2 >= alpha * Fe_1)
        lambda_penalty: penalty weight
        check_rounds: rounds to check for multistep constraint
        popsize: CMA-ES population size
        maxiter: max CMA-ES generations
        sigma0: initial step size
        seed: random seed
        verbose: print progress

    Returns:
        best_params: (N_depth, 4) optimized circuit parameters
        best_loss: final loss value
        info: dict with optimization details
    """
    import cma

    N_l = 2 ** N_depth
    n_params = N_depth * 4
    d_half = float(jnp.sqrt(jnp.pi / 2))

    # Build appropriate loss function
    if constraint_type == "penalty":
        loss_fn = make_stability_loss_penalty(
            logical_0, logical_1, gamma, N_depth,
            alpha_stability=alpha_stability,
            lambda_penalty=lambda_penalty,
        )
    elif constraint_type == "ratio":
        loss_fn = make_stability_loss_ratio(
            logical_0, logical_1, gamma, N_depth,
            alpha_stability=alpha_stability,
            lambda_penalty=lambda_penalty,
        )
    elif constraint_type == "multistep":
        loss_fn = make_stability_loss_multistep(
            logical_0, logical_1, gamma, N_depth,
            alpha_stability=alpha_stability,
            lambda_penalty=lambda_penalty,
            check_rounds=check_rounds,
        )
    else:
        raise ValueError(f"Unknown constraint_type: {constraint_type}")

    def unpack(x_real):
        """Convert real parameter vector to complex (N_depth, 4) params."""
        p = jnp.zeros((N_depth, 4), dtype=jnp.complex64)
        for i in range(N_depth):
            p = p.at[i, 0].set(x_real[4*i] + 1j * x_real[4*i+1])
            p = p.at[i, 1].set(x_real[4*i+2])
            p = p.at[i, 2].set(x_real[4*i+3])
        return p

    def objective(x):
        return loss_fn(unpack(np.array(x)))

    # GKP-informed initial point
    x0 = np.zeros(n_params)
    x0[0] = d_half
    x0[3] = np.pi / 2
    if N_depth > 1:
        x0[5] = d_half
        x0[7] = np.pi / 2

    # Compute baseline Fe_id
    params_zero = jnp.zeros((N_depth, 4), dtype=jnp.complex64)
    alpha_zero, beta_zero = g(params_zero, N_l)
    Fe_id = float(entanglement_fidelity_displacement(
        alpha_zero, beta_zero,
        logical_0.cs, logical_0.ds,
        logical_1.cs, logical_1.ds, gamma))

    if verbose:
        print(f"CMA-ES Lyapunov ({constraint_type}): N_depth={N_depth}, N_l={N_l}, "
              f"params={n_params}, pop={popsize}")
        print(f"  alpha_stability={alpha_stability}, lambda_penalty={lambda_penalty}")
        print(f"  Fe_id={Fe_id:.6f}")
        sys.stdout.flush()

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'maxiter': maxiter, 'popsize': popsize,
        'verbose': -1, 'seed': seed, 'tolfun': 1e-9,
    })

    gen = 0
    best_ever_loss = float('inf')
    t_start = time.time()

    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)
        best_now_loss = es.result.fbest
        best_ever_loss = min(best_ever_loss, best_now_loss)

        if verbose and gen % 100 == 0:
            elapsed = time.time() - t_start
            # Compute actual Fe for current best
            params_best = unpack(es.result.xbest)
            alpha_best, beta_best = g(params_best, N_l)
            Fe_best = float(entanglement_fidelity_displacement(
                alpha_best, beta_best,
                logical_0.cs, logical_0.ds,
                logical_1.cs, logical_1.ds, gamma))
            print(f"  gen {gen}: loss={best_now_loss:.6f}, Fe={Fe_best:.6f} [{elapsed:.0f}s]")
            sys.stdout.flush()
        gen += 1

    elapsed = time.time() - t_start
    best_params = unpack(es.result.xbest)

    # Compute final metrics
    alpha_final, beta_final = g(best_params, N_l)
    Fe_final = float(entanglement_fidelity_displacement(
        alpha_final, beta_final,
        logical_0.cs, logical_0.ds,
        logical_1.cs, logical_1.ds, gamma))

    if verbose:
        print(f"  Done ({elapsed:.0f}s, {gen} gens): loss={es.result.fbest:.6f}, "
              f"Fe={Fe_final:.6f}")
        sys.stdout.flush()

    return best_params, es.result.fbest, {
        'Fe_id': Fe_id,
        'Fe_final': Fe_final,
        'generations': gen,
        'elapsed': elapsed,
        'constraint_type': constraint_type,
        'alpha_stability': alpha_stability,
        'lambda_penalty': lambda_penalty,
    }


def optimize_cmaes_unconstrained(
    logical_0, logical_1, gamma,
    N_depth=6,
    popsize=80,
    maxiter=2000,
    sigma0=3.0,
    seed=42,
    verbose=True,
):
    """
    Standard unconstrained CMA-ES optimization for comparison.

    Maximizes Fe_1 without any stability constraint.
    """
    import cma

    N_l = 2 ** N_depth
    n_params = N_depth * 4
    d_half = float(jnp.sqrt(jnp.pi / 2))

    @jax.jit
    def eval_circuit(p_complex):
        alpha, beta = g(p_complex, N_l)
        return entanglement_fidelity_displacement(
            alpha, beta,
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds, gamma)

    # JIT warmup
    _ = eval_circuit(jnp.zeros((N_depth, 4), dtype=jnp.complex64))

    def unpack(x_real):
        p = jnp.zeros((N_depth, 4), dtype=jnp.complex64)
        for i in range(N_depth):
            p = p.at[i, 0].set(x_real[4*i] + 1j * x_real[4*i+1])
            p = p.at[i, 1].set(x_real[4*i+2])
            p = p.at[i, 2].set(x_real[4*i+3])
        return p

    def objective(x):
        return -float(eval_circuit(unpack(np.array(x))))

    # GKP-informed initial point
    x0 = np.zeros(n_params)
    x0[0] = d_half
    x0[3] = np.pi / 2
    if N_depth > 1:
        x0[5] = d_half
        x0[7] = np.pi / 2

    Fe_id = float(eval_circuit(jnp.zeros((N_depth, 4), dtype=jnp.complex64)))

    if verbose:
        print(f"CMA-ES Unconstrained: N_depth={N_depth}, N_l={N_l}, "
              f"params={n_params}, pop={popsize}")
        print(f"  Fe_id={Fe_id:.6f}")
        sys.stdout.flush()

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'maxiter': maxiter, 'popsize': popsize,
        'verbose': -1, 'seed': seed, 'tolfun': 1e-9,
    })

    gen = 0
    best_ever = 0.0
    t_start = time.time()

    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)
        best_now = -es.result.fbest
        best_ever = max(best_ever, best_now)

        if verbose and gen % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  gen {gen}: Fe={best_now:.6f} (ever={best_ever:.6f}) [{elapsed:.0f}s]")
            sys.stdout.flush()
        gen += 1

    best_fe = -es.result.fbest
    elapsed = time.time() - t_start
    best_params = unpack(es.result.xbest)

    if verbose:
        print(f"  Done ({elapsed:.0f}s, {gen} gens): Fe={best_fe:.6f}, "
              f"improvement={best_fe-Fe_id:.6f}")
        sys.stdout.flush()

    return best_params, best_fe, {
        'Fe_id': Fe_id,
        'generations': gen,
        'elapsed': elapsed,
    }


# ============================================================
# COMPARISON ANALYSIS
# ============================================================

def compare_optimization_methods(
    gamma=0.05,
    Delta=0.3,
    N_trunc=3,
    N_depth=6,
    alpha_stability=0.95,
    lambda_penalty=10.0,
    popsize=80,
    maxiter=1500,
    n_trajectory_rounds=20,
    verbose=True,
):
    """
    Compare Lyapunov-constrained vs unconstrained optimization.

    Returns comprehensive comparison including:
      - Single-round Fe
      - Multi-round Fe trajectory
      - Stability metrics
    """
    print("=" * 74)
    print("  Lyapunov-Constrained vs Unconstrained Optimization Comparison")
    print("=" * 74)
    print(f"  gamma={gamma}, Delta={Delta}, N_depth={N_depth}")
    print(f"  alpha_stability={alpha_stability}, lambda_penalty={lambda_penalty}")
    print()

    # Build GKP states
    logical_0, logical_1 = build_gkp_states(Delta=Delta, N_trunc=N_trunc)

    results = {}

    # 1. Unconstrained optimization
    print("\n--- Unconstrained CMA-ES ---")
    params_unc, fe_unc, info_unc = optimize_cmaes_unconstrained(
        logical_0, logical_1, gamma,
        N_depth=N_depth, popsize=popsize, maxiter=maxiter,
        verbose=verbose,
    )
    results['unconstrained'] = {
        'params': params_unc,
        'Fe_1': fe_unc,
        'info': info_unc,
    }

    # 2. Penalty-constrained optimization
    print("\n--- Penalty-Constrained CMA-ES ---")
    params_pen, loss_pen, info_pen = optimize_cmaes_lyapunov(
        logical_0, logical_1, gamma,
        N_depth=N_depth,
        constraint_type="penalty",
        alpha_stability=alpha_stability,
        lambda_penalty=lambda_penalty,
        popsize=popsize, maxiter=maxiter,
        verbose=verbose,
    )
    results['penalty'] = {
        'params': params_pen,
        'Fe_1': info_pen['Fe_final'],
        'info': info_pen,
    }

    # 3. Ratio-constrained optimization
    print("\n--- Ratio-Constrained CMA-ES ---")
    params_rat, loss_rat, info_rat = optimize_cmaes_lyapunov(
        logical_0, logical_1, gamma,
        N_depth=N_depth,
        constraint_type="ratio",
        alpha_stability=alpha_stability,
        lambda_penalty=lambda_penalty,
        popsize=popsize, maxiter=maxiter,
        verbose=verbose,
    )
    results['ratio'] = {
        'params': params_rat,
        'Fe_1': info_rat['Fe_final'],
        'info': info_rat,
    }

    # 4. Multi-step constrained optimization
    print("\n--- Multi-step Constrained CMA-ES ---")
    params_ms, loss_ms, info_ms = optimize_cmaes_lyapunov(
        logical_0, logical_1, gamma,
        N_depth=N_depth,
        constraint_type="multistep",
        alpha_stability=alpha_stability,
        lambda_penalty=lambda_penalty / 2,  # Lower penalty since summed over rounds
        check_rounds=[1, 2, 3, 5],
        popsize=popsize, maxiter=maxiter,
        verbose=verbose,
    )
    results['multistep'] = {
        'params': params_ms,
        'Fe_1': info_ms['Fe_final'],
        'info': info_ms,
    }

    # Compute Fe trajectories for all methods
    print("\n--- Computing Fe Trajectories ---")
    methods = ['unconstrained', 'penalty', 'ratio', 'multistep']
    for method in methods:
        print(f"  {method}...", end=" ")
        sys.stdout.flush()
        trajectory = compute_fe_trajectory(
            results[method]['params'],
            logical_0, logical_1, gamma, N_depth,
            n_rounds=n_trajectory_rounds,
        )
        results[method]['trajectory'] = trajectory
        print("done.")

    # Also compute identity trajectory
    print("  identity...", end=" ")
    sys.stdout.flush()
    params_id = jnp.zeros((N_depth, 4), dtype=jnp.complex64)
    trajectory_id = compute_fe_trajectory(
        params_id, logical_0, logical_1, gamma, N_depth,
        n_rounds=n_trajectory_rounds,
    )
    results['identity'] = {'trajectory': trajectory_id}
    print("done.")

    # Print comparison summary
    print("\n" + "=" * 74)
    print("  RESULTS SUMMARY")
    print("=" * 74)

    print("\n  Single-round Fe:")
    print(f"    {'Method':<15} {'Fe_1':>10}")
    print(f"    {'-'*15} {'-'*10}")
    print(f"    {'identity':<15} {results['identity']['trajectory'][0]:>10.6f}")
    for method in methods:
        print(f"    {method:<15} {results[method]['Fe_1']:>10.6f}")

    print("\n  Multi-round Fe trajectory:")
    rounds_to_show = [1, 2, 3, 5, 10, 15, 20]
    rounds_to_show = [r for r in rounds_to_show if r <= n_trajectory_rounds]

    header = f"    {'Method':<15}"
    for r in rounds_to_show:
        header += f" {'Fe_'+str(r):>8}"
    print(header)
    print(f"    {'-'*15}" + " " + "-"*8 * len(rounds_to_show))

    all_methods = ['identity'] + methods
    for method in all_methods:
        row = f"    {method:<15}"
        for r in rounds_to_show:
            row += f" {results[method]['trajectory'][r-1]:>8.4f}"
        print(row)

    # Compute stability metrics
    print("\n  Stability Metrics (Fe_n / Fe_1):")
    print(f"    {'Method':<15} {'Fe_2/Fe_1':>10} {'Fe_5/Fe_1':>10} {'Fe_10/Fe_1':>10}")
    print(f"    {'-'*15} {'-'*10} {'-'*10} {'-'*10}")

    for method in methods:
        traj = results[method]['trajectory']
        fe_1 = traj[0]
        ratios = []
        for r in [2, 5, 10]:
            if r <= n_trajectory_rounds:
                ratios.append(traj[r-1] / max(fe_1, 1e-6))
            else:
                ratios.append(float('nan'))
        print(f"    {method:<15} {ratios[0]:>10.4f} {ratios[1]:>10.4f} {ratios[2]:>10.4f}")

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Lyapunov-constrained CD+R optimization")
    parser.add_argument("--gamma", type=float, default=0.05,
                        help="Loss parameter")
    parser.add_argument("--alpha", type=float, default=0.95,
                        help="Stability coefficient (Fe_2 >= alpha * Fe_1)")
    parser.add_argument("--lambda-penalty", type=float, default=10.0,
                        help="Penalty weight for constraint violation")
    parser.add_argument("--N-depth", type=int, default=6,
                        help="CD+R circuit depth")
    parser.add_argument("--popsize", type=int, default=80,
                        help="CMA-ES population size")
    parser.add_argument("--maxiter", type=int, default=1500,
                        help="CMA-ES max iterations")
    parser.add_argument("--rounds", type=int, default=20,
                        help="Number of rounds for trajectory")
    parser.add_argument("--save-dir", type=str, default="results",
                        help="Directory to save results")
    args = parser.parse_args()

    results = compare_optimization_methods(
        gamma=args.gamma,
        N_depth=args.N_depth,
        alpha_stability=args.alpha,
        lambda_penalty=args.lambda_penalty,
        popsize=args.popsize,
        maxiter=args.maxiter,
        n_trajectory_rounds=args.rounds,
        verbose=True,
    )

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    save_dict = {
        'gamma': args.gamma,
        'alpha_stability': args.alpha,
        'lambda_penalty': args.lambda_penalty,
        'N_depth': args.N_depth,
    }
    for method in ['unconstrained', 'penalty', 'ratio', 'multistep']:
        if 'params' in results[method]:
            save_dict[f'params_{method}'] = np.array(results[method]['params'])
        save_dict[f'trajectory_{method}'] = np.array(results[method]['trajectory'])
    save_dict['trajectory_identity'] = np.array(results['identity']['trajectory'])

    npz_path = os.path.join(args.save_dir, f"lyapunov_results_gamma_{args.gamma:.2f}.npz")
    np.savez(npz_path, **save_dict)
    print(f"\n  Results saved to: {npz_path}")


if __name__ == "__main__":
    main()
