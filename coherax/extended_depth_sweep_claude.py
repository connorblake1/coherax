"""
extended_depth_sweep_claude.py

Extended depth sweep from 5 to 10 with multiple strategies to avoid identity basin.

Key findings from initial sweep:
- N_depth=5 achieved BEST Fe=0.869 despite fewer parameters
- Depth 8+ struggles with identity basin (too many parameters)
- Warm-starting helps at some depths but not others

New strategies:
1. Multiple random seeds (BIPOP-style)
2. Warm-start from BEST solution found (not just previous depth)
3. SBS-informed initialization at GKP lattice spacing
4. Track both Direct Fidelity and Fixed-Point objectives

Usage:
    python -m coherax.extended_depth_sweep_claude
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import sys
import os

from coherax.characteristic_jax_utils import (
    gkp_coherent_dm, g, channel_from_b,
    make_pureloss_fock, dqdag, GKP_N,
)
from coherax.transpose_channel_claude import (
    build_gkp_states, coherent_ket_to_fock,
    entanglement_fidelity, entanglement_fidelity_no_recovery,
)
from coherax.coherent_tree_optimizer_claude import (
    entanglement_fidelity_displacement,
)


# Configuration
GAMMA = 0.05
DELTA = 0.3
N_TRUNC = 3
DEPTHS = [5, 6, 7, 8, 9, 10]
POPSIZE = 120  # Larger population for deeper circuits
MAXITER_BASE = 1000  # Base iterations, scale with depth
N_SEEDS = 4  # More seeds for robustness
SIGMA0 = 2.5


def sbs_informed_init(N_depth):
    """
    Initialize parameters inspired by SBS recovery.

    SBS uses displacements at multiples of sqrt(pi/2) in the q/p directions.
    """
    d_half = float(jnp.sqrt(jnp.pi / 2))
    params = np.zeros((N_depth, 4), dtype=np.complex64)

    for i in range(N_depth):
        if i % 2 == 0:
            # Even layers: q-direction displacement
            params[i, 0] = d_half * (1 + 0.1j * (i//2))
            params[i, 2] = np.pi / 2  # theta = pi/2
        else:
            # Odd layers: p-direction displacement
            params[i, 0] = d_half * (0.1 + 1j)
            params[i, 2] = 0  # theta = 0

    return jnp.array(params, dtype=jnp.complex64)


def random_init(N_depth, seed, scale=2.0):
    """Random initialization with controlled scale."""
    rng = np.random.RandomState(seed)
    d_half = float(jnp.sqrt(jnp.pi / 2))

    params = np.zeros((N_depth, 4), dtype=np.complex64)
    for i in range(N_depth):
        params[i, 0] = d_half * (rng.randn() + 1j * rng.randn()) * scale / (i + 1)
        params[i, 1] = rng.randn() * np.pi
        params[i, 2] = rng.randn() * np.pi

    return jnp.array(params, dtype=jnp.complex64)


def extend_params_from_best(best_params, N_depth_new):
    """
    Extend parameters from best-so-far solution to new depth.

    Strategy: Replicate structure with decreasing amplitude.
    """
    N_depth_old = best_params.shape[0]
    if N_depth_new <= N_depth_old:
        return best_params[:N_depth_new]

    params_new = jnp.zeros((N_depth_new, 4), dtype=jnp.complex64)
    params_new = params_new.at[:N_depth_old].set(best_params)

    # For new layers, use scaled-down copies of early layers
    for i in range(N_depth_old, N_depth_new):
        src_idx = (i - N_depth_old) % N_depth_old
        scale = 0.5 / (i - N_depth_old + 1)
        params_new = params_new.at[i, 0].set(best_params[src_idx, 0] * scale)
        params_new = params_new.at[i, 1].set(best_params[src_idx, 1])
        params_new = params_new.at[i, 2].set(best_params[src_idx, 2] + 0.1)

    return params_new


def optimize_cmaes_extended(
    logical_0, logical_1, gamma,
    N_depth, init_params=None, init_type="random",
    popsize=120, maxiter=1500, sigma0=2.5,
    seed=42, verbose=True,
):
    """
    CMA-ES optimization with adaptive parameters based on depth.
    """
    import cma

    N_l = 2 ** N_depth
    n_params = N_depth * 4

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
            x[4*i] = float(p_complex[i, 0].real)
            x[4*i+1] = float(p_complex[i, 0].imag)
            x[4*i+2] = float(p_complex[i, 1].real)
            x[4*i+3] = float(p_complex[i, 2].real)
        return x

    @jax.jit
    def eval_circuit(p_complex):
        alpha, beta = g(p_complex, N_l)
        return entanglement_fidelity_displacement(
            alpha, beta,
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds, gamma)

    # JIT warmup
    _ = eval_circuit(jnp.zeros((N_depth, 4), dtype=jnp.complex64))

    def objective(x):
        return -float(eval_circuit(unpack(np.array(x))))

    # Initialize
    if init_params is not None:
        x0 = pack(init_params)
    else:
        x0 = pack(sbs_informed_init(N_depth))

    Fe_id = float(eval_circuit(jnp.zeros((N_depth, 4), dtype=jnp.complex64)))
    Fe_init = -objective(x0)

    if verbose:
        print(f"  N_depth={N_depth}, N_l={N_l}, params={n_params}")
        print(f"  Init: {init_type}, Fe_init={Fe_init:.6f}, Fe_id={Fe_id:.6f}")
        sys.stdout.flush()

    # Adaptive sigma: smaller for warm-start, larger for deep circuits
    if init_params is not None:
        sigma = sigma0 * 0.3
    else:
        sigma = sigma0 * (1.0 + 0.1 * (N_depth - 5))  # Scale up for deeper

    # Adaptive maxiter: more iterations for deeper circuits
    actual_maxiter = int(maxiter * (1.0 + 0.2 * (N_depth - 5)))

    es = cma.CMAEvolutionStrategy(x0, sigma, {
        'maxiter': actual_maxiter, 'popsize': popsize,
        'verbose': -1, 'seed': seed, 'tolfun': 1e-9,
    })

    gen = 0
    best_ever = Fe_init
    t_start = time.time()

    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)
        best_now = -es.result.fbest
        best_ever = max(best_ever, best_now)

        if verbose and gen % 300 == 0:
            elapsed = time.time() - t_start
            print(f"    gen {gen}: Fe={best_now:.6f} (best={best_ever:.6f}) [{elapsed:.0f}s]")
            sys.stdout.flush()
        gen += 1

    best_fe = -es.result.fbest
    elapsed = time.time() - t_start
    best_params = unpack(es.result.xbest)

    if verbose:
        print(f"  Done ({elapsed:.0f}s): Fe={best_fe:.6f}, improvement={best_fe-Fe_id:+.6f}")
        sys.stdout.flush()

    return best_params, best_fe, {
        'Fe_id': Fe_id,
        'Fe_init': Fe_init,
        'generations': gen,
        'elapsed': elapsed,
        'init_type': init_type,
    }


def run_extended_depth_sweep(gamma=GAMMA, depths=DEPTHS, n_seeds=N_SEEDS, verbose=True):
    """
    Run extended depth sweep with multiple initialization strategies.
    """
    # Build GKP states
    logical_0, logical_1 = build_gkp_states(Delta=DELTA, N_trunc=N_TRUNC)
    psi_0 = coherent_ket_to_fock(logical_0, GKP_N)
    psi_1 = coherent_ket_to_fock(logical_1, GKP_N)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(dqdag(psi_0) @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(dqdag(psi_1) @ psi_1).squeeze())

    # Baselines
    loss_ops = make_pureloss_fock(gamma, rank=10, N=GKP_N)
    Fe_none = float(entanglement_fidelity_no_recovery(loss_ops, psi_0, psi_1))

    if verbose:
        print("=" * 70)
        print(f"Extended Depth Sweep: gamma={gamma}, depths={depths}")
        print(f"Fe(none)={Fe_none:.6f}")
        print(f"n_seeds={n_seeds}, popsize={POPSIZE}")
        print("=" * 70)
        sys.stdout.flush()

    results = {
        'gamma': gamma,
        'Fe_none': Fe_none,
        'depths': [],
        'Fe_best': [],
        'best_method': [],
        'all_results': {},
        'params_best': {},
    }

    # Track best solution across all depths for warm-starting
    global_best_params = None
    global_best_fe = 0
    global_best_depth = None

    for depth in depths:
        if verbose:
            print(f"\n{'='*60}")
            print(f"N_depth = {depth} (N_l = {2**depth}, params = {4*depth})")
            print(f"{'='*60}")
            sys.stdout.flush()

        depth_results = []
        best_depth_fe = 0
        best_depth_params = None
        best_depth_method = None

        # Strategy 1: Multiple cold-start seeds with SBS-informed init
        for seed in range(n_seeds):
            if verbose:
                print(f"\n--- Cold-start seed {seed} (SBS-informed) ---")
                sys.stdout.flush()

            init_p = sbs_informed_init(depth)
            params, fe, info = optimize_cmaes_extended(
                logical_0, logical_1, gamma,
                N_depth=depth, init_params=init_p, init_type="SBS-informed",
                popsize=POPSIZE, maxiter=MAXITER_BASE, sigma0=SIGMA0,
                seed=seed, verbose=verbose,
            )

            depth_results.append(('cold', seed, fe, params))
            if fe > best_depth_fe:
                best_depth_fe = fe
                best_depth_params = params
                best_depth_method = f"cold-seed{seed}"

        # Strategy 2: Random initialization (1 seed) to explore different basins
        if verbose:
            print(f"\n--- Random init ---")
            sys.stdout.flush()

        init_p = random_init(depth, seed=999, scale=2.0)
        params, fe, info = optimize_cmaes_extended(
            logical_0, logical_1, gamma,
            N_depth=depth, init_params=init_p, init_type="random",
            popsize=POPSIZE, maxiter=MAXITER_BASE, sigma0=SIGMA0,
            seed=999, verbose=verbose,
        )

        depth_results.append(('random', 999, fe, params))
        if fe > best_depth_fe:
            best_depth_fe = fe
            best_depth_params = params
            best_depth_method = "random"

        # Strategy 3: Warm-start from global best (if available)
        if global_best_params is not None:
            if verbose:
                print(f"\n--- Warm-start from depth {global_best_depth} (Fe={global_best_fe:.6f}) ---")
                sys.stdout.flush()

            init_p = extend_params_from_best(global_best_params, depth)
            params, fe, info = optimize_cmaes_extended(
                logical_0, logical_1, gamma,
                N_depth=depth, init_params=init_p, init_type="warm-from-best",
                popsize=POPSIZE, maxiter=MAXITER_BASE, sigma0=SIGMA0,
                seed=42, verbose=verbose,
            )

            depth_results.append(('warm-best', global_best_depth, fe, params))
            if fe > best_depth_fe:
                best_depth_fe = fe
                best_depth_params = params
                best_depth_method = f"warm-from-depth{global_best_depth}"

        # Update results
        results['depths'].append(depth)
        results['Fe_best'].append(best_depth_fe)
        results['best_method'].append(best_depth_method)
        results['all_results'][depth] = depth_results
        results['params_best'][depth] = np.array(best_depth_params)

        # Update global best
        if best_depth_fe > global_best_fe:
            global_best_fe = best_depth_fe
            global_best_params = best_depth_params
            global_best_depth = depth

        if verbose:
            print(f"\n  Summary depth {depth}:")
            print(f"    Best Fe: {best_depth_fe:.6f} ({best_depth_method})")
            print(f"    Improvement over none: +{best_depth_fe - Fe_none:.4f}")
            print(f"    Global best: Fe={global_best_fe:.6f} at depth {global_best_depth}")
            sys.stdout.flush()

    return results


def print_summary(results):
    """Print summary table."""
    print("\n" + "=" * 80)
    print("EXTENDED DEPTH SWEEP SUMMARY")
    print("=" * 80)
    print(f"gamma = {results['gamma']}, Fe(none) = {results['Fe_none']:.6f}")
    print("-" * 80)
    print(f"{'Depth':>6} | {'N_l':>6} | {'Params':>6} | {'Best Fe':>10} | {'Δ':>8} | {'Method':<20}")
    print("-" * 80)

    best_overall_fe = 0
    best_overall_depth = None

    for i, depth in enumerate(results['depths']):
        N_l = 2 ** depth
        n_params = 4 * depth
        fe = results['Fe_best'][i]
        delta = fe - results['Fe_none']
        method = results['best_method'][i]

        if fe > best_overall_fe:
            best_overall_fe = fe
            best_overall_depth = depth

        marker = " *" if fe == max(results['Fe_best']) else ""
        print(f"{depth:6d} | {N_l:6d} | {n_params:6d} | {fe:10.6f} | {delta:+8.4f} | {method:<20}{marker}")

    print("-" * 80)
    print(f"BEST: N_depth={best_overall_depth}, Fe={best_overall_fe:.6f}, Δ=+{best_overall_fe - results['Fe_none']:.4f}")
    print("=" * 80)


def save_results(results, filepath='results/extended_depth_sweep.npz'):
    """Save results to NPZ."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    save_dict = {
        'gamma': results['gamma'],
        'Fe_none': results['Fe_none'],
        'depths': np.array(results['depths']),
        'Fe_best': np.array(results['Fe_best']),
        'best_method': np.array(results['best_method'], dtype=object),
    }
    for depth, params in results['params_best'].items():
        save_dict[f'params_depth_{depth}'] = params

    np.savez(filepath, **save_dict)
    print(f"\nSaved to {filepath}")


if __name__ == "__main__":
    # Run for depths 5-10
    results = run_extended_depth_sweep(
        gamma=0.05,
        depths=[5, 6, 7, 8, 9, 10],
        n_seeds=3,
        verbose=True
    )
    print_summary(results)
    save_results(results)
