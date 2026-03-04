"""
multiround_optimizer_claude.py

Multi-round objective optimization for CD+R recovery circuits.

Key idea: Instead of optimizing Fe after 1 round of (loss -> recovery),
optimize for N rounds: Fe_N = Fe((R o E)^N).

This better reflects the long-term behavior of recovery circuits in
continuous stabilization scenarios where cumulative errors compound
over many rounds.

Supports two objective formulations:
1. Direct N-round fidelity: maximize Fe_N
2. Weighted sum: minimize L = sum_n w_n * (1 - Fe_n) for n=1,...,N

The optimizer uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
which is essential for escaping local minima in this multimodal landscape.

Usage:
    python -m gkp_utils.multiround_optimizer_claude
    python -m gkp_utils.multiround_optimizer_claude --rounds 10 --gamma 0.05
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
from jaxtyping import Array

from gkp_utils.characteristic_jax_utils import (
    CoherentKet,
    g,
    channel_from_b,
    make_pureloss_fock,
    apply_kraus_map_nonorm,
    dag,
    dqdag,
    GKP_N,
    dqcoherent,
    gkp_coherent_dm,
    aOmegab,
    coherent_overlap,
)
from gkp_utils.transpose_channel_claude import (
    build_gkp_states,
    coherent_ket_to_fock,
)
from gkp_utils.coherent_tree_optimizer_claude import (
    entanglement_fidelity_displacement,
)


# ============================================================
# MULTI-ROUND FIDELITY IN FOCK BASIS
# ============================================================

def multi_round_fe_fock(recovery_ops, loss_ops, psi_0, psi_1, n_rounds):
    """
    Compute entanglement fidelity after n_rounds of (loss -> recovery).

    For entanglement fidelity, we track the 4 density matrix elements
    rho_{mu,nu} = (R o E)^n (|mu><nu|) independently.

    Args:
        recovery_ops: (K_R, N, N) recovery Kraus operators
        loss_ops: (K_E, N, N) loss Kraus operators
        psi_0, psi_1: (N, 1) normalized Fock kets
        n_rounds: number of (loss -> recovery) cycles

    Returns:
        Fe: entanglement fidelity after n_rounds (real scalar)
    """
    psi = [psi_0, psi_1]

    # Initialize: rho_{mu,nu} = |mu><nu|
    rhos = {}
    for mu in range(2):
        for nu in range(2):
            rhos[(mu, nu)] = psi[mu] @ dag(psi[nu])

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
            Fe += (dag(psi[mu]) @ rhos[(mu, nu)] @ psi[nu]).squeeze()

    return float(jnp.real(Fe) / 4.0)


def multi_round_fe_history_fock(recovery_ops, loss_ops, psi_0, psi_1, n_rounds):
    """
    Compute entanglement fidelity history over n_rounds.

    Returns:
        fe_history: list of Fe values, length n_rounds
    """
    psi = [psi_0, psi_1]

    rhos = {}
    for mu in range(2):
        for nu in range(2):
            rhos[(mu, nu)] = psi[mu] @ dag(psi[nu])

    fe_history = []
    for _ in range(n_rounds):
        for mu in range(2):
            for nu in range(2):
                rho_after_loss = apply_kraus_map_nonorm(loss_ops, rhos[(mu, nu)])
                rhos[(mu, nu)] = apply_kraus_map_nonorm(recovery_ops, rho_after_loss)

        # Compute Fe for this round
        Fe = 0.0 + 0j
        for mu in range(2):
            for nu in range(2):
                Fe += (dag(psi[mu]) @ rhos[(mu, nu)] @ psi[nu]).squeeze()
        fe_history.append(float(jnp.real(Fe) / 4.0))

    return fe_history


# ============================================================
# MULTI-ROUND FIDELITY IN COHERENT BASIS (ANALYTIC)
# ============================================================

def multi_round_fe_coherent_single(alpha, beta, c_0, d_0, c_1, d_1, gamma, n_rounds):
    """
    Compute multi-round entanglement fidelity analytically in coherent basis.

    This computes Fe after n_rounds of (loss -> recovery) by iteratively
    applying the loss channel (which acts simply on coherent states) and
    the recovery channel (represented as displacement operators).

    Note: This is an approximation that tracks only the first-order
    contribution; for more accurate results use the Fock-basis version.

    For a single round, Fe is computed as in entanglement_fidelity_displacement().
    For multiple rounds, we iterate the channel composition.

    Args:
        alpha: (n_kraus, N_disp) complex Kraus coefficients
        beta: (n_kraus, N_disp) complex displacement positions
        c_0, d_0: logical |0> coefficients and positions
        c_1, d_1: logical |1> coefficients and positions
        gamma: loss parameter
        n_rounds: number of (loss -> recovery) cycles

    Returns:
        Fe: entanglement fidelity after n_rounds (real scalar)
    """
    # For n_rounds=1, use the efficient single-round formula
    if n_rounds == 1:
        return entanglement_fidelity_displacement(
            alpha, beta, c_0, d_0, c_1, d_1, gamma
        )

    # For multiple rounds, we need to synthesize to Fock and iterate
    # This is because the coherent-basis formula assumes specific structure
    # that doesn't compose cleanly across rounds

    # Build Fock-basis representations
    N = GKP_N
    recovery_ops = channel_from_b(alpha, beta)
    loss_ops = make_pureloss_fock(gamma, rank=10, N=N)

    # Build Fock logical states
    A0 = c_0.shape[0]
    all_ds = jnp.concatenate([d_0, d_1])
    fock_all = jnp.squeeze(
        jax.vmap(lambda a: dqcoherent(N, a))(all_ds)
    )
    if fock_all.ndim == 3:
        fock_all = fock_all.squeeze(-1)

    psi_0 = jnp.einsum('bn,b->n', fock_all[:A0], c_0).reshape(-1, 1)
    psi_1 = jnp.einsum('bn,b->n', fock_all[A0:], c_1).reshape(-1, 1)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(dag(psi_0) @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(dag(psi_1) @ psi_1).squeeze())

    return multi_round_fe_fock(recovery_ops, loss_ops, psi_0, psi_1, n_rounds)


# ============================================================
# MULTI-ROUND CMA-ES OPTIMIZER
# ============================================================

def optimize_multiround_cmaes(
    logical_0,
    logical_1,
    gamma,
    n_rounds=5,
    N_depth=6,
    popsize=80,
    maxiter=2000,
    sigma0=3.0,
    seed=42,
    verbose=True,
    use_weighted_sum=False,
    weight_decay=0.9,
):
    """
    CMA-ES optimization for multi-round recovery objective.

    Instead of optimizing Fe after 1 round, optimizes Fe after n_rounds
    of (loss -> recovery). This produces circuits that maintain higher
    fidelity over longer stabilization sequences.

    Args:
        logical_0, logical_1: CoherentKet GKP logical states
        gamma: loss parameter (per round)
        n_rounds: number of (loss -> recovery) cycles to optimize for
        N_depth: CD+R circuit depth (N_l = 2^N_depth displacements)
        popsize: CMA-ES population size
        maxiter: maximum CMA-ES generations
        sigma0: initial step size
        seed: random seed
        verbose: print progress
        use_weighted_sum: if True, minimize L = sum_n w_n*(1-Fe_n)
                         if False, maximize Fe after n_rounds
        weight_decay: for weighted sum, w_n = weight_decay^(n-1)

    Returns:
        best_params: (N_depth, 4) optimized circuit parameters
        best_Fe: best final-round entanglement fidelity
        info: dict with optimization details and Fe history
    """
    import cma

    N_l = 2 ** N_depth
    n_params = N_depth * 4
    d_half = float(jnp.sqrt(jnp.pi / 2))
    N = GKP_N

    # Build Fock-basis logical states once
    psi_0 = coherent_ket_to_fock(logical_0, N)
    psi_1 = coherent_ket_to_fock(logical_1, N)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(dag(psi_0) @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(dag(psi_1) @ psi_1).squeeze())

    # Build loss operators once
    loss_ops = make_pureloss_fock(gamma, rank=10, N=N)

    # Compute weights for weighted sum objective
    if use_weighted_sum:
        weights = jnp.array([weight_decay ** n for n in range(n_rounds)])
        weights = weights / jnp.sum(weights)  # Normalize
    else:
        weights = None

    def unpack(x_real):
        """Convert real parameter vector to complex (N_depth, 4) params."""
        p = jnp.zeros((N_depth, 4), dtype=jnp.complex64)
        for i in range(N_depth):
            p = p.at[i, 0].set(x_real[4*i] + 1j * x_real[4*i+1])
            p = p.at[i, 1].set(x_real[4*i+2])
            p = p.at[i, 2].set(x_real[4*i+3])
        return p

    def eval_multiround(p_complex):
        """Evaluate multi-round Fe in Fock basis."""
        alpha, beta = g(p_complex, N_l)
        recovery_ops = channel_from_b(alpha, beta)

        if use_weighted_sum:
            # Weighted sum of (1 - Fe_n)
            fe_history = multi_round_fe_history_fock(
                recovery_ops, loss_ops, psi_0, psi_1, n_rounds
            )
            loss = sum(weights[n] * (1.0 - fe_history[n]) for n in range(n_rounds))
            return -loss  # Return negative for maximization
        else:
            # Just final round Fe
            return multi_round_fe_fock(
                recovery_ops, loss_ops, psi_0, psi_1, n_rounds
            )

    def objective(x):
        return -float(eval_multiround(unpack(np.array(x))))

    # GKP-informed initial point
    x0 = np.zeros(n_params)
    x0[0] = d_half   # Re(d) for layer 0
    x0[3] = np.pi/2  # theta for layer 0 (balanced measurement)
    if N_depth > 1:
        x0[5] = d_half    # Re(d) for layer 1 (orthogonal direction)
        x0[7] = np.pi/2   # theta for layer 1

    # Compute baseline (identity recovery)
    Fe_id = float(eval_multiround(jnp.zeros((N_depth, 4), dtype=jnp.complex64)))

    if verbose:
        obj_type = "weighted-sum" if use_weighted_sum else f"{n_rounds}-round"
        print(f"Multi-round CMA-ES ({obj_type}): N_depth={N_depth}, N_l={N_l}, "
              f"params={n_params}, pop={popsize}")
        print(f"  gamma={gamma}, n_rounds={n_rounds}")
        print(f"  Fe_id (after {n_rounds} rounds)={Fe_id:.6f}")
        sys.stdout.flush()

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'maxiter': maxiter, 'popsize': popsize,
        'verbose': -1, 'seed': seed, 'tolfun': 1e-9,
    })

    gen = 0
    best_ever = 0.0
    above_baseline_gen = None
    t_start = time.time()

    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)
        best_now = -es.result.fbest
        best_ever = max(best_ever, best_now)

        if above_baseline_gen is None and best_ever > Fe_id + 0.001:
            above_baseline_gen = gen
            if verbose:
                print(f"  ** ABOVE BASELINE at gen {gen}! "
                      f"Fe={best_ever:.6f} **")
                sys.stdout.flush()

        if verbose and gen % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  gen {gen}: Fe={best_now:.6f} "
                  f"(ever={best_ever:.6f}) [{elapsed:.0f}s]")
            sys.stdout.flush()
        gen += 1

    best_fe = -es.result.fbest
    elapsed = time.time() - t_start
    best_params = unpack(es.result.xbest)

    # Compute full Fe history for best solution
    alpha, beta = g(best_params, N_l)
    recovery_ops = channel_from_b(alpha, beta)
    fe_history = multi_round_fe_history_fock(
        recovery_ops, loss_ops, psi_0, psi_1, n_rounds
    )

    if verbose:
        print(f"  Done ({elapsed:.0f}s, {gen} gens): "
              f"Fe[{n_rounds}]={best_fe:.6f}, improvement={best_fe-Fe_id:.6f}")
        print(f"  Fe history: {[f'{fe:.4f}' for fe in fe_history]}")
        sys.stdout.flush()

    return best_params, best_fe, {
        'Fe_id': Fe_id,
        'generations': gen,
        'elapsed': elapsed,
        'above_baseline_gen': above_baseline_gen,
        'xbest': es.result.xbest,
        'n_rounds': n_rounds,
        'fe_history': fe_history,
        'use_weighted_sum': use_weighted_sum,
    }


def bipop_multiround_cmaes(
    logical_0,
    logical_1,
    gamma,
    n_rounds=5,
    N_depth=6,
    n_restarts=10,
    popsize=80,
    maxiter=1000,
    sigma0=3.0,
    verbose=True,
    use_weighted_sum=False,
    weight_decay=0.9,
):
    """
    BIPOP-style CMA-ES with multiple random restarts for multi-round objective.

    The multi-round optimization landscape is highly multimodal. This function
    runs many independent CMA-ES trials with different seeds and returns the
    best result.

    Args:
        logical_0, logical_1: CoherentKet GKP logical states
        gamma: loss parameter
        n_rounds: number of (loss -> recovery) cycles
        N_depth: CD+R circuit depth
        n_restarts: number of independent CMA-ES restarts
        popsize: CMA-ES population size
        maxiter: max generations per restart
        sigma0: initial step size
        verbose: print progress
        use_weighted_sum: if True, use weighted sum objective
        weight_decay: weight decay factor for weighted sum

    Returns:
        best_params: (N_depth, 4) optimized circuit parameters
        best_Fe: best final-round entanglement fidelity
        info: dict with all trial results
    """
    import cma

    N_l = 2 ** N_depth
    d_half = float(jnp.sqrt(jnp.pi / 2))
    N = GKP_N

    # Build Fock-basis logical states once
    psi_0 = coherent_ket_to_fock(logical_0, N)
    psi_1 = coherent_ket_to_fock(logical_1, N)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(dag(psi_0) @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(dag(psi_1) @ psi_1).squeeze())

    # Build loss operators once
    loss_ops = make_pureloss_fock(gamma, rank=10, N=N)

    # Compute weights for weighted sum objective
    if use_weighted_sum:
        weights = jnp.array([weight_decay ** n for n in range(n_rounds)])
        weights = weights / jnp.sum(weights)
    else:
        weights = None

    def unpack(x_real):
        p = jnp.zeros((N_depth, 4), dtype=jnp.complex64)
        for i in range(N_depth):
            p = p.at[i, 0].set(x_real[4*i] + 1j * x_real[4*i+1])
            p = p.at[i, 1].set(x_real[4*i+2])
            p = p.at[i, 2].set(x_real[4*i+3])
        return p

    def eval_multiround(p_complex):
        alpha, beta = g(p_complex, N_l)
        recovery_ops = channel_from_b(alpha, beta)

        if use_weighted_sum:
            fe_history = multi_round_fe_history_fock(
                recovery_ops, loss_ops, psi_0, psi_1, n_rounds
            )
            loss = sum(weights[n] * (1.0 - fe_history[n]) for n in range(n_rounds))
            return -loss
        else:
            return multi_round_fe_fock(
                recovery_ops, loss_ops, psi_0, psi_1, n_rounds
            )

    def objective(x):
        return -float(eval_multiround(unpack(np.array(x))))

    Fe_id = float(eval_multiround(jnp.zeros((N_depth, 4), dtype=jnp.complex64)))

    if verbose:
        obj_type = "weighted-sum" if use_weighted_sum else f"{n_rounds}-round"
        print(f"BIPOP Multi-round CMA-ES ({obj_type}): N_depth={N_depth}, N_l={N_l}, "
              f"restarts={n_restarts}, pop={popsize}")
        print(f"  gamma={gamma}, n_rounds={n_rounds}")
        print(f"  Fe_id (after {n_rounds} rounds)={Fe_id:.6f}")
        sys.stdout.flush()

    best_fe = Fe_id
    best_x = np.zeros(N_depth * 4)
    trials = []
    t_total = time.time()

    for trial in range(n_restarts):
        x0 = np.zeros(N_depth * 4)
        x0[0] = d_half
        x0[3] = np.pi/2
        if N_depth > 1:
            x0[5] = d_half
            x0[7] = np.pi/2

        es = cma.CMAEvolutionStrategy(x0, sigma0, {
            'maxiter': maxiter, 'popsize': popsize,
            'verbose': -1, 'seed': trial, 'tolfun': 1e-9,
        })

        gen = 0
        t0 = time.time()
        while not es.stop():
            solutions = es.ask()
            fitnesses = [objective(x) for x in solutions]
            es.tell(solutions, fitnesses)
            gen += 1

        fe = -es.result.fbest
        elapsed = time.time() - t0
        improved = fe > Fe_id + 0.001

        trials.append({'seed': trial, 'Fe': fe, 'gens': gen, 'time': elapsed})

        if verbose:
            flag = ' ***' if improved else ''
            print(f"  trial {trial:2d}: Fe[{n_rounds}]={fe:.6f} "
                  f"({gen} gens, {elapsed:.0f}s){flag}")
            sys.stdout.flush()

        if fe > best_fe:
            best_fe = fe
            best_x = es.result.xbest.copy()

    elapsed_total = time.time() - t_total
    best_params = unpack(best_x)
    n_improved = sum(1 for t in trials if t['Fe'] > Fe_id + 0.001)

    # Compute full Fe history for best solution
    alpha, beta = g(best_params, N_l)
    recovery_ops = channel_from_b(alpha, beta)
    fe_history = multi_round_fe_history_fock(
        recovery_ops, loss_ops, psi_0, psi_1, n_rounds
    )

    if verbose:
        print(f"\n  Best Fe[{n_rounds}]={best_fe:.6f} (+{best_fe-Fe_id:+.6f})")
        print(f"  Improved: {n_improved}/{n_restarts} trials")
        print(f"  Fe history: {[f'{fe:.4f}' for fe in fe_history]}")
        print(f"  Total time: {elapsed_total:.0f}s")
        sys.stdout.flush()

    return best_params, best_fe, {
        'Fe_id': Fe_id,
        'trials': trials,
        'n_improved': n_improved,
        'total_time': elapsed_total,
        'n_rounds': n_rounds,
        'fe_history': fe_history,
        'use_weighted_sum': use_weighted_sum,
    }


# ============================================================
# COMPARISON: SINGLE-ROUND VS MULTI-ROUND OPTIMIZED
# ============================================================

def compare_single_vs_multiround(
    gamma=0.05,
    Delta=0.3,
    N_trunc=3,
    n_rounds=10,
    N_depth=6,
    popsize=80,
    maxiter=1500,
    seed=42,
    verbose=True,
):
    """
    Compare circuits optimized for single-round vs multi-round objectives.

    Demonstrates that circuits optimized for multi-round fidelity can
    maintain higher fidelity over extended stabilization sequences than
    circuits optimized only for single-round performance.

    Args:
        gamma: loss parameter
        Delta: GKP envelope parameter
        N_trunc: GKP state truncation
        n_rounds: number of rounds to evaluate
        N_depth: CD+R circuit depth
        popsize: CMA-ES population size
        maxiter: max CMA-ES generations
        seed: random seed
        verbose: print progress

    Returns:
        results: dict with comparison data
    """
    N = GKP_N
    N_l = 2 ** N_depth

    if verbose:
        print("=" * 70)
        print("  Single-Round vs Multi-Round Optimization Comparison")
        print("=" * 70)
        print(f"  gamma={gamma}, Delta={Delta}, n_rounds={n_rounds}")
        print(f"  N_depth={N_depth}, N_l={N_l}")
        print()

    # Build GKP states
    if verbose:
        print("Building GKP states...", end=" ")
        sys.stdout.flush()
    logical_0, logical_1 = build_gkp_states(Delta=Delta, N_trunc=N_trunc)
    psi_0 = coherent_ket_to_fock(logical_0, N)
    psi_1 = coherent_ket_to_fock(logical_1, N)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(dag(psi_0) @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(dag(psi_1) @ psi_1).squeeze())
    if verbose:
        print("done.")

    # Build loss operators
    loss_ops = make_pureloss_fock(gamma, rank=10, N=N)

    # Identity (no recovery) baseline
    if verbose:
        print("\n--- Identity (No Recovery) ---")
    identity_history = multi_round_fe_history_fock(
        jnp.eye(N, dtype=jnp.complex64).reshape(1, N, N),
        loss_ops, psi_0, psi_1, n_rounds
    )
    if verbose:
        print(f"  Fe history: {[f'{fe:.4f}' for fe in identity_history[:5]]}...")

    # Single-round optimized circuit
    if verbose:
        print("\n--- Single-Round Optimized ---")
    params_single, fe_single, info_single = optimize_multiround_cmaes(
        logical_0, logical_1, gamma,
        n_rounds=1,  # Optimize for single round
        N_depth=N_depth, popsize=popsize, maxiter=maxiter,
        seed=seed, verbose=verbose,
    )

    # Evaluate single-round optimized over multiple rounds
    alpha_s, beta_s = g(params_single, N_l)
    recovery_single = channel_from_b(alpha_s, beta_s)
    single_history = multi_round_fe_history_fock(
        recovery_single, loss_ops, psi_0, psi_1, n_rounds
    )
    if verbose:
        print(f"  Multi-round Fe: {[f'{fe:.4f}' for fe in single_history[:5]]}...")

    # Multi-round optimized circuit
    if verbose:
        print(f"\n--- Multi-Round Optimized (n={n_rounds}) ---")
    params_multi, fe_multi, info_multi = optimize_multiround_cmaes(
        logical_0, logical_1, gamma,
        n_rounds=n_rounds,  # Optimize for n_rounds
        N_depth=N_depth, popsize=popsize, maxiter=maxiter,
        seed=seed + 1000, verbose=verbose,
    )

    # Evaluate multi-round optimized
    alpha_m, beta_m = g(params_multi, N_l)
    recovery_multi = channel_from_b(alpha_m, beta_m)
    multi_history = multi_round_fe_history_fock(
        recovery_multi, loss_ops, psi_0, psi_1, n_rounds
    )

    # Weighted-sum optimized circuit
    if verbose:
        print(f"\n--- Weighted-Sum Optimized ---")
    params_weighted, fe_weighted, info_weighted = optimize_multiround_cmaes(
        logical_0, logical_1, gamma,
        n_rounds=n_rounds,
        N_depth=N_depth, popsize=popsize, maxiter=maxiter,
        seed=seed + 2000, verbose=verbose,
        use_weighted_sum=True, weight_decay=0.9,
    )

    alpha_w, beta_w = g(params_weighted, N_l)
    recovery_weighted = channel_from_b(alpha_w, beta_w)
    weighted_history = multi_round_fe_history_fock(
        recovery_weighted, loss_ops, psi_0, psi_1, n_rounds
    )

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("  COMPARISON SUMMARY")
        print("=" * 70)
        print(f"\n  Fe at each round:")
        print(f"  {'Round':>5s} | {'Identity':>10s} | {'Single-Opt':>10s} | "
              f"{'Multi-Opt':>10s} | {'Weighted':>10s}")
        print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

        for r in range(min(n_rounds, 15)):
            print(f"  {r+1:5d} | {identity_history[r]:10.6f} | "
                  f"{single_history[r]:10.6f} | {multi_history[r]:10.6f} | "
                  f"{weighted_history[r]:10.6f}")

        print(f"\n  Key observations:")
        print(f"    - Single-round opt achieves Fe[1]={single_history[0]:.6f}")
        print(f"    - Multi-round opt achieves Fe[1]={multi_history[0]:.6f}, "
              f"Fe[{n_rounds}]={multi_history[-1]:.6f}")
        print(f"    - Weighted-sum opt achieves Fe[1]={weighted_history[0]:.6f}, "
              f"Fe[{n_rounds}]={weighted_history[-1]:.6f}")

        # Compare at round 5 and final round
        if n_rounds >= 5:
            print(f"\n  At round 5:")
            print(f"    Single-opt: {single_history[4]:.6f}")
            print(f"    Multi-opt:  {multi_history[4]:.6f} "
                  f"({multi_history[4]-single_history[4]:+.6f})")
            print(f"    Weighted:   {weighted_history[4]:.6f} "
                  f"({weighted_history[4]-single_history[4]:+.6f})")

    return {
        'gamma': gamma,
        'Delta': Delta,
        'n_rounds': n_rounds,
        'N_depth': N_depth,
        'identity_history': identity_history,
        'single_history': single_history,
        'multi_history': multi_history,
        'weighted_history': weighted_history,
        'params_single': params_single,
        'params_multi': params_multi,
        'params_weighted': params_weighted,
        'info_single': info_single,
        'info_multi': info_multi,
        'info_weighted': info_weighted,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-round objective optimization for CD+R recovery")
    parser.add_argument("--gamma", type=float, default=0.05,
                        help="Loss parameter (per round)")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of rounds for multi-round objective")
    parser.add_argument("--n-depth", type=int, default=6,
                        help="CD+R circuit depth")
    parser.add_argument("--popsize", type=int, default=80,
                        help="CMA-ES population size")
    parser.add_argument("--maxiter", type=int, default=1500,
                        help="Max CMA-ES generations")
    parser.add_argument("--compare", action="store_true",
                        help="Run comparison between single and multi-round")
    parser.add_argument("--weighted", action="store_true",
                        help="Use weighted-sum objective")
    parser.add_argument("--save-dir", default="results",
                        help="Directory to save results")
    args = parser.parse_args()

    if args.compare:
        # Run full comparison
        results = compare_single_vs_multiround(
            gamma=args.gamma,
            n_rounds=args.rounds,
            N_depth=args.n_depth,
            popsize=args.popsize,
            maxiter=args.maxiter,
        )

        # Save results
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(
            args.save_dir,
            f"multiround_comparison_gamma{args.gamma:.2f}_N{args.rounds}.npz"
        )
        np.savez(
            save_path,
            gamma=args.gamma,
            n_rounds=args.rounds,
            N_depth=args.n_depth,
            identity_history=np.array(results['identity_history']),
            single_history=np.array(results['single_history']),
            multi_history=np.array(results['multi_history']),
            weighted_history=np.array(results['weighted_history']),
            params_single=np.array(results['params_single']),
            params_multi=np.array(results['params_multi']),
            params_weighted=np.array(results['params_weighted']),
        )
        print(f"\nResults saved to: {save_path}")

    else:
        # Run single optimization
        print("=" * 60)
        print("Multi-Round CD+R Optimizer")
        print("=" * 60)

        logical_0 = gkp_coherent_dm(mu=0, N_trunc=3, Delta=0.3, lattice='square')
        logical_1 = gkp_coherent_dm(mu=1, N_trunc=3, Delta=0.3, lattice='square')

        params, fe, info = optimize_multiround_cmaes(
            logical_0, logical_1,
            gamma=args.gamma,
            n_rounds=args.rounds,
            N_depth=args.n_depth,
            popsize=args.popsize,
            maxiter=args.maxiter,
            use_weighted_sum=args.weighted,
            verbose=True,
        )

        print(f"\n{'='*60}")
        print(f"  Final Results")
        print(f"{'='*60}")
        print(f"  gamma={args.gamma}, n_rounds={args.rounds}")
        print(f"  Fe[{args.rounds}]={fe:.6f}")
        print(f"  Fe history: {info['fe_history']}")

        # Save parameters
        os.makedirs(args.save_dir, exist_ok=True)
        gamma_key = f"{args.gamma:.2f}".replace(".", "p")
        save_path = os.path.join(
            args.save_dir,
            f"multiround_params_gamma{gamma_key}_N{args.rounds}.npz"
        )
        np.savez(
            save_path,
            gamma=args.gamma,
            n_rounds=args.rounds,
            N_depth=args.n_depth,
            params=np.array(params),
            fe_history=np.array(info['fe_history']),
        )
        print(f"\nParameters saved to: {save_path}")


if __name__ == "__main__":
    main()
