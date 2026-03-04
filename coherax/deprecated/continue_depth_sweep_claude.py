"""
continue_depth_sweep_claude.py

Continue the warm-start chain from depth 10 to depths 11-12.

Based on extended_depth_sweep_claude.py results:
- Depth 7 cold: Fe = 0.909
- Depth 8 warm from 7: Fe = 0.921
- Depth 10 warm from 8: Fe = 0.950

Now continue:
- Depth 11 warm from 10
- Depth 12 warm from 10 (or 11 if better)

Usage:
    python -m coherax.continue_depth_sweep_claude
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
POPSIZE = 120
MAXITER_BASE = 1500  # More iterations for deep circuits
SIGMA0 = 2.5


def extend_params(best_params, N_depth_new):
    """
    Extend parameters from previous depth to new depth.

    Strategy: Copy all layers, add new layers with small scaled copies.
    """
    N_depth_old = best_params.shape[0]
    if N_depth_new <= N_depth_old:
        return best_params[:N_depth_new]

    params_new = jnp.zeros((N_depth_new, 4), dtype=jnp.complex64)
    params_new = params_new.at[:N_depth_old].set(best_params)

    # For new layers, use scaled-down copies of early layers
    for i in range(N_depth_old, N_depth_new):
        src_idx = (i - N_depth_old) % N_depth_old
        scale = 0.3 / (i - N_depth_old + 1)
        params_new = params_new.at[i, 0].set(best_params[src_idx, 0] * scale)
        params_new = params_new.at[i, 1].set(best_params[src_idx, 1])
        params_new = params_new.at[i, 2].set(best_params[src_idx, 2] + 0.05)

    return params_new


def optimize_cmaes_deep(
    logical_0, logical_1, gamma,
    N_depth, init_params, init_type="warm",
    popsize=120, maxiter=2000, sigma0=0.8,
    seed=42, verbose=True,
):
    """
    CMA-ES optimization for deep circuits (11-12).
    Uses smaller sigma for warm-start and more iterations.
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

    x0 = pack(init_params)
    Fe_id = float(eval_circuit(jnp.zeros((N_depth, 4), dtype=jnp.complex64)))
    Fe_init = -objective(x0)

    if verbose:
        print(f"  N_depth={N_depth}, N_l={N_l}, params={n_params}")
        print(f"  Init: {init_type}, Fe_init={Fe_init:.6f}, Fe_id={Fe_id:.6f}")
        sys.stdout.flush()

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'maxiter': maxiter, 'popsize': popsize,
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


def run_continuation(source_depth=10, target_depths=[11, 12], verbose=True):
    """
    Continue warm-start chain from source_depth to target_depths.
    """
    # Load existing results
    data = np.load('results/extended_depth_sweep.npz', allow_pickle=True)

    gamma = float(data['gamma'])
    Fe_none = float(data['Fe_none'])

    # Load source parameters
    source_params = jnp.array(data[f'params_depth_{source_depth}'])
    source_fe = float(data['Fe_best'][list(data['depths']).index(source_depth)])

    if verbose:
        print("=" * 70)
        print(f"Continuing Depth Sweep: {source_depth} -> {target_depths}")
        print(f"gamma={gamma}, Fe(none)={Fe_none:.6f}")
        print(f"Source: depth {source_depth}, Fe={source_fe:.6f}")
        print("=" * 70)
        sys.stdout.flush()

    # Build GKP states
    logical_0, logical_1 = build_gkp_states(Delta=DELTA, N_trunc=N_TRUNC)

    # Track results
    results = {
        'gamma': gamma,
        'Fe_none': Fe_none,
        'source_depth': source_depth,
        'source_fe': source_fe,
        'depths': [],
        'Fe_best': [],
        'params_best': {},
    }

    current_best_params = source_params
    current_best_fe = source_fe
    current_best_depth = source_depth

    for depth in target_depths:
        if verbose:
            print(f"\n{'='*60}")
            print(f"N_depth = {depth} (N_l = {2**depth}, params = {4*depth})")
            print(f"{'='*60}")
            print(f"\n--- Warm-start from depth {current_best_depth} (Fe={current_best_fe:.6f}) ---")
            sys.stdout.flush()

        # Extend parameters
        init_p = extend_params(current_best_params, depth)

        # Run optimization
        params, fe, info = optimize_cmaes_deep(
            logical_0, logical_1, gamma,
            N_depth=depth, init_params=init_p,
            init_type=f"warm-from-depth{current_best_depth}",
            popsize=POPSIZE, maxiter=MAXITER_BASE, sigma0=0.8,
            seed=42, verbose=verbose,
        )

        results['depths'].append(depth)
        results['Fe_best'].append(fe)
        results['params_best'][depth] = np.array(params)

        if verbose:
            delta = fe - Fe_none
            pct = 100 * (fe - Fe_none) / (1.0 - Fe_none)
            print(f"\n  Summary depth {depth}:")
            print(f"    Fe: {fe:.6f}")
            print(f"    Improvement: +{delta:.4f} ({pct:.1f}% of infidelity)")
            sys.stdout.flush()

        # Update best for next iteration
        if fe > current_best_fe:
            current_best_params = params
            current_best_fe = fe
            current_best_depth = depth

    return results


def save_continuation_results(results, existing_path='results/extended_depth_sweep.npz',
                               output_path='results/deep_depth_sweep.npz'):
    """Save continuation results, merging with existing data."""
    # Load existing
    existing = dict(np.load(existing_path, allow_pickle=True))

    # Merge
    depths = list(existing['depths']) + results['depths']
    Fe_best = list(existing['Fe_best']) + results['Fe_best']
    best_method = list(existing['best_method']) + [f"warm-from-depth{results['source_depth']}"] * len(results['depths'])

    save_dict = {
        'gamma': results['gamma'],
        'Fe_none': results['Fe_none'],
        'depths': np.array(depths),
        'Fe_best': np.array(Fe_best),
        'best_method': np.array(best_method, dtype=object),
    }

    # Copy existing params
    for key in existing:
        if key.startswith('params_depth_'):
            save_dict[key] = existing[key]

    # Add new params
    for depth, params in results['params_best'].items():
        save_dict[f'params_depth_{depth}'] = params

    np.savez(output_path, **save_dict)
    print(f"\nSaved to {output_path}")

    # Print final summary
    print("\n" + "=" * 80)
    print("FULL DEPTH SWEEP SUMMARY (5-12)")
    print("=" * 80)
    print(f"gamma = {results['gamma']}, Fe(none) = {results['Fe_none']:.6f}")
    print("-" * 80)
    print(f"{'Depth':>6} | {'N_l':>6} | {'Params':>6} | {'Best Fe':>10} | {'Δ':>8} | {'% Recovered':>12}")
    print("-" * 80)

    for i, depth in enumerate(depths):
        N_l = 2 ** depth
        n_params = 4 * depth
        fe = Fe_best[i]
        delta = fe - results['Fe_none']
        pct = 100 * delta / (1.0 - results['Fe_none'])
        marker = " *" if fe == max(Fe_best) else ""
        print(f"{depth:6d} | {N_l:6d} | {n_params:6d} | {fe:10.6f} | {delta:+8.4f} | {pct:11.1f}%{marker}")

    print("-" * 80)
    best_idx = np.argmax(Fe_best)
    print(f"BEST: N_depth={depths[best_idx]}, Fe={Fe_best[best_idx]:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    # Continue from depth 10 to 11-12
    results = run_continuation(
        source_depth=10,
        target_depths=[11, 12],
        verbose=True
    )
    save_continuation_results(results)
