"""
curriculum_optimizer_claude.py

Curriculum learning for CD+R circuit optimization on GKP error correction.

Key insight: At gamma=0 (no loss), the identity channel is optimal. As gamma
increases, we need to do progressively more active recovery. By gradually
increasing gamma from near-zero to the target value, warm-starting each stage
from the previous optimized solution, we can potentially:
  1. Avoid getting trapped in the identity basin at high gamma
  2. Find solutions that smoothly interpolate from identity to active recovery
  3. Produce more stable multi-round behavior

This addresses the "identity basin problem" observed in direct CMA-ES optimization
at high gamma values, where most random seeds converge to near-identity solutions
that perform worse than the optimal non-trivial recovery.

Usage:
    python -m gkp_utils.curriculum_optimizer_claude
    python -m gkp_utils.curriculum_optimizer_claude --gamma_target 0.10 --N_depth 6
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import sys
import os
import argparse
import json

from gkp_utils.characteristic_jax_utils import (
    g, channel_from_b, make_pureloss_fock, make_transpose_for_pureloss,
    GKP_N, gkp_coherent_dm,
)
from gkp_utils.transpose_channel_claude import (
    build_gkp_states, coherent_ket_to_fock,
    build_sbs_kraus, entanglement_fidelity,
    entanglement_fidelity_no_recovery,
)
from gkp_utils.coherent_tree_optimizer_claude import (
    entanglement_fidelity_displacement,
    optimize_cmaes_flat,
    bipop_cmaes_flat,
)
from gkp_utils.multi_round_claude import simulate_rounds


# ============================================================
# CURRICULUM SCHEDULE BUILDERS
# ============================================================

def build_exponential_schedule(gamma_target, n_stages=10, gamma_min=0.001):
    """
    Build exponentially-spaced gamma schedule from gamma_min to gamma_target.

    Exponential spacing places more stages at low gamma (where identity is
    nearly optimal) and fewer at high gamma. This matches the intuition that
    the solution should change slowly at low gamma but more rapidly as we
    approach the target.

    Args:
        gamma_target: final gamma value
        n_stages: number of curriculum stages
        gamma_min: starting gamma value

    Returns:
        schedule: list of gamma values from gamma_min to gamma_target
    """
    if gamma_target <= gamma_min:
        return [gamma_target]

    log_min = np.log(gamma_min)
    log_max = np.log(gamma_target)
    log_schedule = np.linspace(log_min, log_max, n_stages)
    return list(np.exp(log_schedule))


def build_linear_schedule(gamma_target, n_stages=10, gamma_min=0.001):
    """
    Build linearly-spaced gamma schedule.

    Linear spacing gives equal attention to all gamma ranges.

    Args:
        gamma_target: final gamma value
        n_stages: number of curriculum stages
        gamma_min: starting gamma value

    Returns:
        schedule: list of gamma values from gamma_min to gamma_target
    """
    if gamma_target <= gamma_min:
        return [gamma_target]

    return list(np.linspace(gamma_min, gamma_target, n_stages))


def build_adaptive_schedule(gamma_target, gamma_min=0.001, density="exponential"):
    """
    Build adaptive gamma schedule with more stages at higher gamma.

    The schedule density is computed based on expected difficulty:
    more stages are added where the optimization landscape changes rapidly.

    Args:
        gamma_target: final gamma value
        gamma_min: starting gamma value
        density: "exponential", "linear", or "sqrt"

    Returns:
        schedule: list of gamma values
    """
    if gamma_target <= gamma_min:
        return [gamma_target]

    if density == "exponential":
        # Denser at low gamma
        n_stages = max(5, int(10 * np.log10(gamma_target / gamma_min)))
        return build_exponential_schedule(gamma_target, n_stages, gamma_min)
    elif density == "linear":
        # Uniform density
        n_stages = max(5, int(100 * (gamma_target - gamma_min)))
        return build_linear_schedule(gamma_target, n_stages, gamma_min)
    elif density == "sqrt":
        # Denser at high gamma (sqrt spacing)
        n_stages = max(5, int(10 * np.sqrt(gamma_target / gamma_min)))
        sqrt_min = np.sqrt(gamma_min)
        sqrt_max = np.sqrt(gamma_target)
        sqrt_schedule = np.linspace(sqrt_min, sqrt_max, n_stages)
        return list(sqrt_schedule ** 2)
    else:
        raise ValueError(f"Unknown density: {density}")


# ============================================================
# CURRICULUM OPTIMIZATION
# ============================================================

def curriculum_optimize(
    logical_0, logical_1,
    gamma_target,
    gamma_schedule=None,
    N_depth=6,
    popsize=80,
    maxiter_per_stage=500,
    sigma0=3.0,
    sigma_decay=0.7,
    verbose=True,
):
    """
    Curriculum learning optimization: gradually increase gamma, warm-starting
    each stage from the previous solution.

    Hypothesis: At low gamma, identity is optimal, and the optimization landscape
    smoothly transitions to active recovery as gamma increases. By tracking this
    transition, we avoid discontinuous jumps into bad local minima.

    Args:
        logical_0, logical_1: CoherentKet GKP logical states
        gamma_target: target loss parameter
        gamma_schedule: list of gamma values from low to target, or None for auto
        N_depth: CD+R circuit depth (N_l = 2^N_depth)
        popsize: CMA-ES population size
        maxiter_per_stage: max generations per curriculum stage
        sigma0: initial CMA-ES step size for first stage
        sigma_decay: multiply sigma by this factor each stage
        verbose: print progress

    Returns:
        best_params: (N_depth, 4) optimized circuit parameters
        Fe_final: entanglement fidelity at gamma_target
        history: dict with per-stage results
    """
    import cma

    N_l = 2 ** N_depth
    n_params = N_depth * 4
    d_half = float(jnp.sqrt(jnp.pi / 2))

    # Build gamma schedule if not provided
    if gamma_schedule is None:
        gamma_schedule = build_exponential_schedule(gamma_target, n_stages=10)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Curriculum Optimization")
        print(f"{'='*60}")
        print(f"  gamma_target = {gamma_target}")
        print(f"  N_depth = {N_depth}, N_l = {N_l}")
        print(f"  Schedule: {len(gamma_schedule)} stages")
        print(f"  Gammas: {[f'{g:.4f}' for g in gamma_schedule]}")
        print()
        sys.stdout.flush()

    def unpack(x_real):
        """Convert real parameter vector to complex (N_depth, 4) params."""
        p = jnp.zeros((N_depth, 4), dtype=jnp.complex64)
        for i in range(N_depth):
            p = p.at[i, 0].set(x_real[4*i] + 1j * x_real[4*i+1])
            p = p.at[i, 1].set(x_real[4*i+2])
            p = p.at[i, 2].set(x_real[4*i+3])
        return p

    def pack(p_complex):
        """Convert complex (N_depth, 4) params to real vector."""
        x = np.zeros(n_params)
        for i in range(N_depth):
            x[4*i] = float(np.real(p_complex[i, 0]))
            x[4*i+1] = float(np.imag(p_complex[i, 0]))
            x[4*i+2] = float(np.real(p_complex[i, 1]))
            x[4*i+3] = float(np.real(p_complex[i, 2]))
        return x

    def make_objective(gamma):
        """Create objective function for given gamma."""
        @jax.jit
        def eval_circuit(p_complex):
            alpha, beta = g(p_complex, N_l)
            return entanglement_fidelity_displacement(
                alpha, beta,
                logical_0.cs, logical_0.ds,
                logical_1.cs, logical_1.ds, gamma)
        return lambda x: -float(eval_circuit(unpack(np.array(x))))

    # Initialize near identity with small GKP-informed perturbation
    x_current = np.zeros(n_params)
    x_current[0] = d_half * 0.1  # Small displacement
    x_current[3] = np.pi/4       # Slightly off-center

    history = {
        'gammas': [],
        'Fe_values': [],
        'generations': [],
        'sigmas': [],
        'params': [],
    }

    sigma_current = sigma0
    t_total = time.time()

    for stage_idx, gamma in enumerate(gamma_schedule):
        t0 = time.time()

        objective = make_objective(gamma)

        # Compute Fe at current params for this gamma (before optimization)
        Fe_before = -objective(x_current)
        Fe_id = -objective(np.zeros(n_params))

        if verbose:
            print(f"  Stage {stage_idx+1}/{len(gamma_schedule)}: gamma={gamma:.4f}")
            print(f"    Fe_id={Fe_id:.6f}, Fe_init={Fe_before:.6f}")
            sys.stdout.flush()

        # Run CMA-ES from current solution
        es = cma.CMAEvolutionStrategy(x_current.copy(), sigma_current, {
            'maxiter': maxiter_per_stage,
            'popsize': popsize,
            'verbose': -1,
            'seed': stage_idx,
            'tolfun': 1e-9,
        })

        gen = 0
        while not es.stop():
            solutions = es.ask()
            fitnesses = [objective(x) for x in solutions]
            es.tell(solutions, fitnesses)
            gen += 1

        Fe_after = -es.result.fbest
        x_current = es.result.xbest.copy()
        elapsed = time.time() - t0

        # Record history
        history['gammas'].append(gamma)
        history['Fe_values'].append(Fe_after)
        history['generations'].append(gen)
        history['sigmas'].append(sigma_current)
        history['params'].append(unpack(x_current))

        if verbose:
            improvement = Fe_after - Fe_id
            delta = Fe_after - Fe_before
            print(f"    Fe_final={Fe_after:.6f} (improv={improvement:+.6f}, "
                  f"delta={delta:+.6f})")
            print(f"    {gen} gens, {elapsed:.1f}s, sigma={sigma_current:.3f}")
            sys.stdout.flush()

        # Decay sigma for next stage (solutions get more refined)
        sigma_current *= sigma_decay
        sigma_current = max(sigma_current, 0.1)  # Don't go too small

    elapsed_total = time.time() - t_total

    best_params = unpack(x_current)
    Fe_final = history['Fe_values'][-1]

    if verbose:
        print(f"\n  Curriculum complete in {elapsed_total:.0f}s")
        print(f"  Final Fe at gamma={gamma_target}: {Fe_final:.6f}")
        sys.stdout.flush()

    return best_params, Fe_final, history


def curriculum_optimize_bipop(
    logical_0, logical_1,
    gamma_target,
    gamma_schedule=None,
    N_depth=6,
    n_restarts_first=5,
    n_restarts_later=1,
    popsize=80,
    maxiter_first=1000,
    maxiter_later=500,
    sigma0=3.0,
    verbose=True,
):
    """
    Curriculum with BIPOP at first stage, single-restart warm-start thereafter.

    At gamma_min, we use BIPOP (multiple restarts) to find a good basin.
    For subsequent stages, we use single-restart warm-start optimization.

    Args:
        logical_0, logical_1: CoherentKet GKP logical states
        gamma_target: target loss parameter
        gamma_schedule: list of gamma values, or None for auto
        N_depth: CD+R circuit depth
        n_restarts_first: BIPOP restarts for first stage
        n_restarts_later: restarts per stage after first (usually 1)
        popsize: CMA-ES population size
        maxiter_first: max generations for first stage
        maxiter_later: max generations for later stages
        sigma0: initial step size
        verbose: print progress

    Returns:
        best_params, Fe_final, history
    """
    import cma

    N_l = 2 ** N_depth
    n_params = N_depth * 4
    d_half = float(jnp.sqrt(jnp.pi / 2))

    if gamma_schedule is None:
        gamma_schedule = build_exponential_schedule(gamma_target, n_stages=10)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Curriculum Optimization (BIPOP start)")
        print(f"{'='*60}")
        print(f"  gamma_target = {gamma_target}")
        print(f"  N_depth = {N_depth}, N_l = {N_l}")
        print(f"  Schedule: {len(gamma_schedule)} stages")
        print(f"  First stage: {n_restarts_first} BIPOP restarts")
        print()
        sys.stdout.flush()

    def unpack(x_real):
        p = jnp.zeros((N_depth, 4), dtype=jnp.complex64)
        for i in range(N_depth):
            p = p.at[i, 0].set(x_real[4*i] + 1j * x_real[4*i+1])
            p = p.at[i, 1].set(x_real[4*i+2])
            p = p.at[i, 2].set(x_real[4*i+3])
        return p

    def pack(p_complex):
        x = np.zeros(n_params)
        for i in range(N_depth):
            x[4*i] = float(np.real(p_complex[i, 0]))
            x[4*i+1] = float(np.imag(p_complex[i, 0]))
            x[4*i+2] = float(np.real(p_complex[i, 1]))
            x[4*i+3] = float(np.real(p_complex[i, 2]))
        return x

    history = {
        'gammas': [],
        'Fe_values': [],
        'generations': [],
        'params': [],
    }

    x_current = None
    t_total = time.time()

    for stage_idx, gamma in enumerate(gamma_schedule):
        t0 = time.time()

        if stage_idx == 0:
            # First stage: BIPOP
            if verbose:
                print(f"  Stage 1/{len(gamma_schedule)}: gamma={gamma:.4f} "
                      f"(BIPOP x{n_restarts_first})")
                sys.stdout.flush()

            params, Fe, info = bipop_cmaes_flat(
                logical_0, logical_1, gamma,
                N_depth=N_depth,
                n_restarts=n_restarts_first,
                popsize=popsize,
                maxiter=maxiter_first,
                sigma0=sigma0,
                verbose=verbose,
            )
            x_current = pack(params)

        else:
            # Later stages: warm-start from previous
            if verbose:
                print(f"  Stage {stage_idx+1}/{len(gamma_schedule)}: "
                      f"gamma={gamma:.4f} (warm-start)")
                sys.stdout.flush()

            @jax.jit
            def eval_circuit(p_complex):
                alpha, beta = g(p_complex, N_l)
                return entanglement_fidelity_displacement(
                    alpha, beta,
                    logical_0.cs, logical_0.ds,
                    logical_1.cs, logical_1.ds, gamma)

            objective = lambda x: -float(eval_circuit(unpack(np.array(x))))
            Fe_id = -objective(np.zeros(n_params))

            best_fe = -objective(x_current)
            best_x = x_current.copy()

            for restart in range(n_restarts_later):
                es = cma.CMAEvolutionStrategy(x_current.copy(), sigma0 * 0.5, {
                    'maxiter': maxiter_later,
                    'popsize': popsize,
                    'verbose': -1,
                    'seed': stage_idx * 100 + restart,
                    'tolfun': 1e-9,
                })

                while not es.stop():
                    solutions = es.ask()
                    fitnesses = [objective(x) for x in solutions]
                    es.tell(solutions, fitnesses)

                Fe = -es.result.fbest
                if Fe > best_fe:
                    best_fe = Fe
                    best_x = es.result.xbest.copy()

            x_current = best_x
            params = unpack(x_current)

            if verbose:
                print(f"    Fe={best_fe:.6f} (vs Fe_id={Fe_id:.6f})")
                sys.stdout.flush()

        Fe = float(entanglement_fidelity_displacement(
            *g(unpack(x_current), N_l),
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds, gamma))

        history['gammas'].append(gamma)
        history['Fe_values'].append(Fe)
        history['generations'].append(0)  # Not tracked for BIPOP
        history['params'].append(unpack(x_current))

        elapsed = time.time() - t0
        if verbose:
            print(f"    Stage time: {elapsed:.1f}s")
            sys.stdout.flush()

    elapsed_total = time.time() - t_total

    best_params = unpack(x_current)
    Fe_final = history['Fe_values'][-1]

    if verbose:
        print(f"\n  Curriculum complete in {elapsed_total:.0f}s")
        print(f"  Final Fe at gamma={gamma_target}: {Fe_final:.6f}")
        sys.stdout.flush()

    return best_params, Fe_final, history


# ============================================================
# COMPARISON METHODS
# ============================================================

def direct_optimize(
    logical_0, logical_1, gamma,
    N_depth=6, popsize=80, maxiter=2000, sigma0=3.0, seed=42, verbose=True,
):
    """Direct CMA-ES optimization at target gamma (no curriculum)."""
    params, Fe, info = optimize_cmaes_flat(
        logical_0, logical_1, gamma,
        N_depth=N_depth, popsize=popsize, maxiter=maxiter,
        sigma0=sigma0, seed=seed, verbose=verbose,
    )
    return params, Fe, info


def random_restarts_optimize(
    logical_0, logical_1, gamma,
    N_depth=6, n_restarts=10, popsize=80, maxiter=1000, sigma0=3.0, verbose=True,
):
    """BIPOP-style random restarts at target gamma."""
    params, Fe, info = bipop_cmaes_flat(
        logical_0, logical_1, gamma,
        N_depth=N_depth, n_restarts=n_restarts, popsize=popsize,
        maxiter=maxiter, sigma0=sigma0, verbose=verbose,
    )
    return params, Fe, info


# ============================================================
# MULTI-ROUND STABILITY EVALUATION
# ============================================================

def evaluate_multi_round_stability(
    params, logical_0, logical_1, gamma,
    n_rounds=20, N_depth=6, loss_rank=10, verbose=True,
):
    """
    Evaluate multi-round entanglement fidelity stability.

    Good solutions should maintain high fidelity over many rounds of
    (loss -> recovery). Unstable solutions may show rapid degradation.

    Args:
        params: (N_depth, 4) circuit parameters
        logical_0, logical_1: CoherentKet states
        gamma: loss parameter per round
        n_rounds: number of (loss -> recovery) cycles
        N_depth: circuit depth
        loss_rank: Kraus rank for loss channel
        verbose: print progress

    Returns:
        fe_history: list of Fe values after each round
        stability_metric: Fe at round n_rounds / Fe at round 1
    """
    N_l = 2 ** N_depth

    # Build Fock-basis operators
    alpha, beta = g(params, N_l)
    recovery_ops = channel_from_b(alpha, beta)
    loss_ops = make_pureloss_fock(gamma, rank=loss_rank)

    # Build Fock kets
    psi_0 = coherent_ket_to_fock(logical_0)
    psi_1 = coherent_ket_to_fock(logical_1)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(jnp.conj(psi_0).T @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(jnp.conj(psi_1).T @ psi_1).squeeze())

    # Simulate
    fe_history = simulate_rounds(recovery_ops, loss_ops, psi_0, psi_1, n_rounds)

    # Stability metric
    stability = fe_history[-1] / fe_history[0] if fe_history[0] > 0 else 0.0

    if verbose:
        print(f"  Multi-round stability (gamma={gamma}, {n_rounds} rounds):")
        print(f"    Fe[1] = {fe_history[0]:.6f}")
        print(f"    Fe[{n_rounds}] = {fe_history[-1]:.6f}")
        print(f"    Stability ratio: {stability:.4f}")
        sys.stdout.flush()

    return fe_history, stability


# ============================================================
# MAIN COMPARISON
# ============================================================

def run_curriculum_comparison(
    gamma_targets=None,
    N_depth=6,
    n_rounds_eval=20,
    save_dir="results",
    verbose=True,
):
    """
    Compare curriculum learning vs direct/random-restart optimization.

    For each target gamma:
      1. Curriculum learning (exponential schedule)
      2. Direct CMA-ES at target
      3. BIPOP random restarts at target

    Evaluate both single-round Fe and multi-round stability.
    """
    if gamma_targets is None:
        gamma_targets = [0.05, 0.10, 0.15]

    print("=" * 70)
    print("  Curriculum Learning Comparison")
    print("=" * 70)
    print(f"  gamma_targets: {gamma_targets}")
    print(f"  N_depth: {N_depth}")
    print(f"  Multi-round eval: {n_rounds_eval} rounds")
    print()

    # Build GKP states
    print("Building GKP states...", end=" ")
    sys.stdout.flush()
    Delta = 0.3
    N_trunc = 3
    logical_0 = gkp_coherent_dm(mu=0, N_trunc=N_trunc, Delta=Delta, lattice='square')
    logical_1 = gkp_coherent_dm(mu=1, N_trunc=N_trunc, Delta=Delta, lattice='square')
    print("done.")

    # Also build Fock kets for multi-round
    psi_0 = coherent_ket_to_fock(logical_0)
    psi_1 = coherent_ket_to_fock(logical_1)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(jnp.conj(psi_0).T @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(jnp.conj(psi_1).T @ psi_1).squeeze())

    all_results = {}

    for gamma_target in gamma_targets:
        print(f"\n{'='*70}")
        print(f"  Target gamma = {gamma_target}")
        print(f"{'='*70}")

        gamma_results = {}

        # Compute identity baseline
        Fe_id = float(entanglement_fidelity_displacement(
            jnp.ones((1, 1), dtype=jnp.complex64),
            jnp.zeros((1, 1), dtype=jnp.complex64),
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds, gamma_target))

        # Compute transpose bound (Fock)
        loss_ops = make_pureloss_fock(gamma_target, rank=10)
        transpose_ops = make_transpose_for_pureloss(loss_ops, logical_0, logical_1)
        Fe_transpose = float(entanglement_fidelity(
            transpose_ops, loss_ops, psi_0, psi_1))

        print(f"  Fe_id = {Fe_id:.6f}")
        print(f"  Fe_transpose = {Fe_transpose:.6f}")

        # ----- 1. Curriculum Learning -----
        print(f"\n--- Method 1: Curriculum Learning ---")
        t0 = time.time()
        params_curr, Fe_curr, hist_curr = curriculum_optimize(
            logical_0, logical_1, gamma_target,
            N_depth=N_depth, popsize=80, maxiter_per_stage=500,
            verbose=verbose,
        )
        time_curr = time.time() - t0

        # Multi-round evaluation
        fe_hist_curr, stab_curr = evaluate_multi_round_stability(
            params_curr, logical_0, logical_1, gamma_target,
            n_rounds=n_rounds_eval, N_depth=N_depth, verbose=verbose,
        )

        gamma_results['curriculum'] = {
            'params': np.array(params_curr),
            'Fe': Fe_curr,
            'time': time_curr,
            'fe_history': fe_hist_curr,
            'stability': stab_curr,
            'curriculum_history': {k: v if k != 'params' else [np.array(p) for p in v]
                                   for k, v in hist_curr.items()},
        }

        # ----- 2. Direct CMA-ES -----
        print(f"\n--- Method 2: Direct CMA-ES ---")
        t0 = time.time()
        params_dir, Fe_dir, info_dir = direct_optimize(
            logical_0, logical_1, gamma_target,
            N_depth=N_depth, popsize=80, maxiter=2000,
            verbose=verbose,
        )
        time_dir = time.time() - t0

        fe_hist_dir, stab_dir = evaluate_multi_round_stability(
            params_dir, logical_0, logical_1, gamma_target,
            n_rounds=n_rounds_eval, N_depth=N_depth, verbose=verbose,
        )

        gamma_results['direct'] = {
            'params': np.array(params_dir),
            'Fe': Fe_dir,
            'time': time_dir,
            'fe_history': fe_hist_dir,
            'stability': stab_dir,
        }

        # ----- 3. Random Restarts -----
        print(f"\n--- Method 3: BIPOP Random Restarts ---")
        t0 = time.time()
        params_bipop, Fe_bipop, info_bipop = random_restarts_optimize(
            logical_0, logical_1, gamma_target,
            N_depth=N_depth, n_restarts=10, popsize=80, maxiter=1000,
            verbose=verbose,
        )
        time_bipop = time.time() - t0

        fe_hist_bipop, stab_bipop = evaluate_multi_round_stability(
            params_bipop, logical_0, logical_1, gamma_target,
            n_rounds=n_rounds_eval, N_depth=N_depth, verbose=verbose,
        )

        gamma_results['bipop'] = {
            'params': np.array(params_bipop),
            'Fe': Fe_bipop,
            'time': time_bipop,
            'fe_history': fe_hist_bipop,
            'stability': stab_bipop,
        }

        # Store baselines
        gamma_results['Fe_id'] = Fe_id
        gamma_results['Fe_transpose'] = Fe_transpose

        all_results[gamma_target] = gamma_results

        # Print comparison
        print(f"\n  {'='*60}")
        print(f"  Comparison at gamma={gamma_target}")
        print(f"  {'='*60}")
        print(f"  {'Method':<15s} | {'Fe':>10s} | {'Stability':>10s} | {'Time':>8s}")
        print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
        print(f"  {'Identity':<15s} | {Fe_id:10.6f} | {'---':>10s} | {'---':>8s}")
        print(f"  {'Transpose':<15s} | {Fe_transpose:10.6f} | {'---':>10s} | {'---':>8s}")
        print(f"  {'Curriculum':<15s} | {Fe_curr:10.6f} | {stab_curr:10.4f} | "
              f"{time_curr:7.0f}s")
        print(f"  {'Direct':<15s} | {Fe_dir:10.6f} | {stab_dir:10.4f} | "
              f"{time_dir:7.0f}s")
        print(f"  {'BIPOP':<15s} | {Fe_bipop:10.6f} | {stab_bipop:10.4f} | "
              f"{time_bipop:7.0f}s")
        sys.stdout.flush()

    # ============================================================
    # OVERALL SUMMARY
    # ============================================================
    print(f"\n\n{'='*70}")
    print("  OVERALL SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Single-round Fe:")
    print(f"  {'gamma':>8s} | {'Fe_id':>10s} | {'Curriculum':>10s} | "
          f"{'Direct':>10s} | {'BIPOP':>10s} | {'Transpose':>10s}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for gamma in gamma_targets:
        r = all_results[gamma]
        print(f"  {gamma:8.3f} | {r['Fe_id']:10.6f} | "
              f"{r['curriculum']['Fe']:10.6f} | {r['direct']['Fe']:10.6f} | "
              f"{r['bipop']['Fe']:10.6f} | {r['Fe_transpose']:10.6f}")

    print(f"\n  Multi-round Stability (Fe[{n_rounds_eval}]/Fe[1]):")
    print(f"  {'gamma':>8s} | {'Curriculum':>10s} | {'Direct':>10s} | {'BIPOP':>10s}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for gamma in gamma_targets:
        r = all_results[gamma]
        print(f"  {gamma:8.3f} | {r['curriculum']['stability']:10.4f} | "
              f"{r['direct']['stability']:10.4f} | {r['bipop']['stability']:10.4f}")

    print(f"\n  Final Fe after {n_rounds_eval} rounds:")
    print(f"  {'gamma':>8s} | {'Curriculum':>10s} | {'Direct':>10s} | {'BIPOP':>10s}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for gamma in gamma_targets:
        r = all_results[gamma]
        print(f"  {gamma:8.3f} | {r['curriculum']['fe_history'][-1]:10.6f} | "
              f"{r['direct']['fe_history'][-1]:10.6f} | "
              f"{r['bipop']['fe_history'][-1]:10.6f}")

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    os.makedirs(save_dir, exist_ok=True)

    # Save to NPZ
    save_dict = {
        'gamma_targets': np.array(gamma_targets),
        'N_depth': N_depth,
        'n_rounds_eval': n_rounds_eval,
    }
    for gamma in gamma_targets:
        g_key = f"{gamma:.2f}".replace(".", "p")
        r = all_results[gamma]
        save_dict[f'Fe_id_{g_key}'] = r['Fe_id']
        save_dict[f'Fe_transpose_{g_key}'] = r['Fe_transpose']
        for method in ['curriculum', 'direct', 'bipop']:
            save_dict[f'Fe_{method}_{g_key}'] = r[method]['Fe']
            save_dict[f'params_{method}_{g_key}'] = r[method]['params']
            save_dict[f'fe_history_{method}_{g_key}'] = np.array(r[method]['fe_history'])
            save_dict[f'stability_{method}_{g_key}'] = r[method]['stability']
            save_dict[f'time_{method}_{g_key}'] = r[method]['time']

    npz_path = os.path.join(save_dir, "curriculum_comparison.npz")
    np.savez(npz_path, **save_dict)
    print(f"\n  Results saved to: {npz_path}")

    # Save JSON summary (without large arrays)
    json_summary = {}
    for gamma in gamma_targets:
        r = all_results[gamma]
        json_summary[str(gamma)] = {
            'Fe_id': r['Fe_id'],
            'Fe_transpose': r['Fe_transpose'],
            'curriculum': {
                'Fe': r['curriculum']['Fe'],
                'stability': r['curriculum']['stability'],
                'time': r['curriculum']['time'],
                'fe_final': r['curriculum']['fe_history'][-1],
            },
            'direct': {
                'Fe': r['direct']['Fe'],
                'stability': r['direct']['stability'],
                'time': r['direct']['time'],
                'fe_final': r['direct']['fe_history'][-1],
            },
            'bipop': {
                'Fe': r['bipop']['Fe'],
                'stability': r['bipop']['stability'],
                'time': r['bipop']['time'],
                'fe_final': r['bipop']['fe_history'][-1],
            },
        }

    json_path = os.path.join(save_dir, "curriculum_comparison.json")
    with open(json_path, 'w') as f:
        json.dump(json_summary, f, indent=2)
    print(f"  Summary saved to: {json_path}")

    return all_results


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Curriculum learning for CD+R optimization")
    parser.add_argument("--gamma_target", type=float, default=None,
                        help="Single target gamma (default: run comparison)")
    parser.add_argument("--gammas", type=float, nargs="+",
                        default=None,
                        help="Target gammas for comparison")
    parser.add_argument("--N_depth", type=int, default=6,
                        help="CD+R circuit depth")
    parser.add_argument("--rounds", type=int, default=20,
                        help="Multi-round evaluation rounds")
    parser.add_argument("--save_dir", default="results",
                        help="Output directory")
    args = parser.parse_args()

    if args.gamma_target is not None:
        # Single gamma: just run curriculum
        print("=" * 70)
        print(f"  Single Curriculum Run: gamma_target={args.gamma_target}")
        print("=" * 70)

        Delta = 0.3
        N_trunc = 3
        logical_0 = gkp_coherent_dm(mu=0, N_trunc=N_trunc, Delta=Delta, lattice='square')
        logical_1 = gkp_coherent_dm(mu=1, N_trunc=N_trunc, Delta=Delta, lattice='square')

        params, Fe, history = curriculum_optimize(
            logical_0, logical_1, args.gamma_target,
            N_depth=args.N_depth, verbose=True,
        )

        fe_hist, stab = evaluate_multi_round_stability(
            params, logical_0, logical_1, args.gamma_target,
            n_rounds=args.rounds, N_depth=args.N_depth, verbose=True,
        )

        print(f"\n  Final: Fe={Fe:.6f}, stability={stab:.4f}")

    else:
        # Full comparison
        gamma_targets = args.gammas if args.gammas else [0.05, 0.10, 0.15]
        run_curriculum_comparison(
            gamma_targets=gamma_targets,
            N_depth=args.N_depth,
            n_rounds_eval=args.rounds,
            save_dir=args.save_dir,
            verbose=True,
        )


if __name__ == "__main__":
    main()
