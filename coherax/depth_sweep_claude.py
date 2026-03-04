"""
depth_sweep_claude.py

Sweep circuit depths from 5 to 10 with warm-starting:
- Use optimized params at depth N to initialize depth N+1
- Compare Fixed-Point (superoperator) vs Direct Fidelity (CMA-ES) objectives
- Checkpoint results at each depth

Inspired by channel_construction checkpointing where intermediate states
(e.g., x-lattice GKP) are used to bootstrap full construction.

Usage:
    python -m coherax.depth_sweep_claude
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
GAMMA = 0.05  # Focus on one gamma for depth sweep
DELTA = 0.3
N_TRUNC = 3
DEPTHS = [5, 6, 7, 8, 9, 10]
POPSIZE = 100  # Larger for deeper circuits
MAXITER = 1500  # Per depth
SIGMA0 = 2.0


def extend_params(params_small, N_depth_new):
    """
    Extend parameters from depth N to depth N+1.
    
    Strategy: Copy existing layers, add a small perturbation layer at the end.
    The new layer is initialized to approximately identity (small displacement).
    """
    N_depth_old = params_small.shape[0]
    if N_depth_new <= N_depth_old:
        return params_small[:N_depth_new]
    
    # Create extended params
    params_new = jnp.zeros((N_depth_new, 4), dtype=jnp.complex64)
    params_new = params_new.at[:N_depth_old].set(params_small)
    
    # Initialize new layers with small random perturbations
    # Small displacement, balanced theta
    for i in range(N_depth_old, N_depth_new):
        scale = 0.1 / (i - N_depth_old + 1)  # Decreasing scale for later layers
        params_new = params_new.at[i, 0].set(scale * (1.0 + 1.0j))
        params_new = params_new.at[i, 2].set(jnp.pi / 4)  # Moderate theta
    
    return params_new


def optimize_cmaes_warmstart(
    logical_0, logical_1, gamma,
    N_depth, init_params=None,
    popsize=100, maxiter=1500, sigma0=2.0,
    seed=42, verbose=True,
):
    """
    CMA-ES optimization with optional warm-start from previous depth.
    """
    import cma
    
    N_l = 2 ** N_depth
    n_params = N_depth * 4
    d_half = float(jnp.sqrt(jnp.pi / 2))
    
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
        # Warm-start: extend previous params
        extended = extend_params(init_params, N_depth)
        x0 = pack(extended)
        init_type = "warm-start"
    else:
        # GKP-informed init
        x0 = np.zeros(n_params)
        x0[0] = d_half
        x0[3] = np.pi/2
        if N_depth > 1:
            x0[5] = d_half
            x0[7] = np.pi/2
        init_type = "GKP-informed"
    
    Fe_id = float(eval_circuit(jnp.zeros((N_depth, 4), dtype=jnp.complex64)))
    Fe_init = -objective(x0)
    
    if verbose:
        print(f"  N_depth={N_depth}, N_l={N_l}, params={n_params}")
        print(f"  Init: {init_type}, Fe_init={Fe_init:.6f}, Fe_id={Fe_id:.6f}")
        sys.stdout.flush()
    
    # Adaptive sigma based on whether warm-starting
    sigma = sigma0 * 0.5 if init_params is not None else sigma0
    
    es = cma.CMAEvolutionStrategy(x0, sigma, {
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
        
        if verbose and gen % 200 == 0:
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


def run_depth_sweep(gamma=GAMMA, depths=DEPTHS, n_seeds=3, verbose=True):
    """
    Run depth sweep with warm-starting between depths.
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
        print(f"Depth Sweep: gamma={gamma}, depths={depths}")
        print(f"Fe(none)={Fe_none:.6f}")
        print("=" * 70)
        sys.stdout.flush()
    
    results = {
        'gamma': gamma,
        'Fe_none': Fe_none,
        'depths': [],
        'Fe_cold': [],      # Cold-start (no warm-start)
        'Fe_warm': [],      # Warm-start from previous depth
        'params_best': {},
        'timing': [],
    }
    
    prev_params = None
    
    for depth in depths:
        if verbose:
            print(f"\n{'='*50}")
            print(f"N_depth = {depth}")
            print(f"{'='*50}")
            sys.stdout.flush()
        
        N_l = 2 ** depth
        
        # Run multiple seeds for cold-start
        best_cold_fe = 0
        best_cold_params = None
        
        for seed in range(n_seeds):
            if verbose:
                print(f"\n--- Cold-start seed {seed} ---")
                sys.stdout.flush()
            
            params, fe, info = optimize_cmaes_warmstart(
                logical_0, logical_1, gamma,
                N_depth=depth, init_params=None,
                popsize=POPSIZE, maxiter=MAXITER, sigma0=SIGMA0,
                seed=seed, verbose=verbose,
            )
            
            if fe > best_cold_fe:
                best_cold_fe = fe
                best_cold_params = params
        
        # Warm-start from previous depth (if available)
        if prev_params is not None:
            if verbose:
                print(f"\n--- Warm-start from depth {depth-1} ---")
                sys.stdout.flush()
            
            params_warm, fe_warm, info_warm = optimize_cmaes_warmstart(
                logical_0, logical_1, gamma,
                N_depth=depth, init_params=prev_params,
                popsize=POPSIZE, maxiter=MAXITER, sigma0=SIGMA0,
                seed=42, verbose=verbose,
            )
        else:
            fe_warm = best_cold_fe
            params_warm = best_cold_params
        
        # Use best overall
        if fe_warm > best_cold_fe:
            best_params = params_warm
            best_fe = fe_warm
        else:
            best_params = best_cold_params
            best_fe = best_cold_fe
        
        results['depths'].append(depth)
        results['Fe_cold'].append(best_cold_fe)
        results['Fe_warm'].append(fe_warm)
        results['params_best'][depth] = np.array(best_params)
        
        # Update prev_params for next depth
        prev_params = best_params
        
        if verbose:
            print(f"\n  Summary depth {depth}:")
            print(f"    Best cold-start: Fe={best_cold_fe:.6f}")
            print(f"    Warm-start:      Fe={fe_warm:.6f}")
            print(f"    Improvement over none: +{best_fe - Fe_none:.4f}")
            sys.stdout.flush()
    
    return results


def print_summary(results):
    """Print summary table."""
    print("\n" + "=" * 70)
    print("DEPTH SWEEP SUMMARY")
    print("=" * 70)
    print(f"gamma = {results['gamma']}, Fe(none) = {results['Fe_none']:.6f}")
    print("-" * 70)
    print(f"{'Depth':>6} | {'N_l':>6} | {'Fe(cold)':>10} | {'Fe(warm)':>10} | {'Best':>10} | {'Δ':>8}")
    print("-" * 70)
    
    for i, depth in enumerate(results['depths']):
        N_l = 2 ** depth
        fe_cold = results['Fe_cold'][i]
        fe_warm = results['Fe_warm'][i]
        fe_best = max(fe_cold, fe_warm)
        delta = fe_best - results['Fe_none']
        winner = "warm" if fe_warm > fe_cold else "cold"
        print(f"{depth:6d} | {N_l:6d} | {fe_cold:10.6f} | {fe_warm:10.6f} | "
              f"{fe_best:10.6f} | {delta:+8.4f} ({winner})")
    
    print("=" * 70)


def save_results(results, filepath='results/depth_sweep_results.npz'):
    """Save results to NPZ."""
    save_dict = {
        'gamma': results['gamma'],
        'Fe_none': results['Fe_none'],
        'depths': np.array(results['depths']),
        'Fe_cold': np.array(results['Fe_cold']),
        'Fe_warm': np.array(results['Fe_warm']),
    }
    for depth, params in results['params_best'].items():
        save_dict[f'params_depth_{depth}'] = params
    
    np.savez(filepath, **save_dict)
    print(f"\nSaved to {filepath}")


if __name__ == "__main__":
    results = run_depth_sweep(gamma=0.05, depths=[5, 6, 7, 8], n_seeds=2, verbose=True)
    print_summary(results)
    save_results(results)
