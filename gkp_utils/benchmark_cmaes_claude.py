"""
benchmark_cmaes_claude.py

Benchmark CMA-ES recovery vs SBS, transpose channel, and identity baseline.
Saves results and optimized parameters to NPZ file for reproducibility.

Usage:
    python -m gkp_utils.benchmark_cmaes_claude
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import sys

from gkp_utils.characteristic_jax_utils import (
    gkp_coherent_dm, g, channel_from_b,
    make_pureloss_fock, make_transpose_for_pureloss,
    apply_kraus_map_nonorm, dqcoherent, dqdag, GKP_N,
)
from gkp_utils.transpose_channel_claude import (
    build_gkp_states, coherent_ket_to_fock,
    build_sbs_kraus, entanglement_fidelity,
    entanglement_fidelity_no_recovery,
)
from gkp_utils.coherent_tree_optimizer_claude import (
    entanglement_fidelity_displacement,
    optimize_cmaes_flat,
)


# Configuration
GAMMAS = [0.03, 0.05, 0.07, 0.1, 0.15]
BEST_SEEDS = {
    0.03: 789,
    0.05: 1,
    0.07: 123,
    0.10: 42,
    0.15: 1,
}
N_DEPTH = 6
N_L = 2 ** N_DEPTH
DELTA = 0.3
N_TRUNC = 3
POPSIZE = 80
SIGMA0 = 3.0
MAXITER = 2000


def run_benchmark(gammas=GAMMAS, verbose=True):
    """
    Run full benchmark comparing CMA-ES vs baselines.

    Returns:
        results: dict with all fidelity values and parameters
    """
    # Build GKP states
    logical_0, logical_1 = build_gkp_states(Delta=DELTA, N_trunc=N_TRUNC)
    psi_0 = coherent_ket_to_fock(logical_0, GKP_N)
    psi_1 = coherent_ket_to_fock(logical_1, GKP_N)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(dqdag(psi_0) @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(dqdag(psi_1) @ psi_1).squeeze())

    results = {
        'gammas': np.array(gammas),
        'Fe_none': [],
        'Fe_sbs': [],
        'Fe_cmaes': [],
        'Fe_transpose': [],
        'Fe_cmaes_fock': [],
        'best_seeds': [],
        'params': {},
        'metadata': {
            'N_depth': N_DEPTH,
            'N_l': N_L,
            'Delta': DELTA,
            'N_trunc': N_TRUNC,
            'popsize': POPSIZE,
            'sigma0': SIGMA0,
            'lattice': 'square',
        }
    }

    if verbose:
        print("=" * 70)
        print("CMA-ES Recovery Benchmark")
        print("=" * 70)
        print(f"N_depth={N_DEPTH}, N_l={N_L}, Delta={DELTA}, N_trunc={N_TRUNC}")
        print(f"Gammas: {gammas}")
        print("=" * 70)
        sys.stdout.flush()

    for gamma in gammas:
        if verbose:
            print(f"\n--- gamma = {gamma} ---")
            sys.stdout.flush()

        # Loss channel
        loss_ops = make_pureloss_fock(gamma, rank=10, N=GKP_N)

        # 1. No recovery baseline
        Fe_none = float(entanglement_fidelity_no_recovery(loss_ops, psi_0, psi_1))
        results['Fe_none'].append(Fe_none)
        if verbose:
            print(f"  Fe (none):      {Fe_none:.6f}")
            sys.stdout.flush()

        # 2. SBS recovery
        sbs_ops = build_sbs_kraus(Delta=DELTA, N=GKP_N)
        Fe_sbs = float(entanglement_fidelity(sbs_ops, loss_ops, psi_0, psi_1))
        results['Fe_sbs'].append(Fe_sbs)
        if verbose:
            print(f"  Fe (SBS):       {Fe_sbs:.6f}")
            sys.stdout.flush()

        # 3. Transpose channel (theoretical optimum)
        transpose_ops = make_transpose_for_pureloss(loss_ops, logical_0, logical_1)
        Fe_transpose = float(entanglement_fidelity(transpose_ops, loss_ops, psi_0, psi_1))
        results['Fe_transpose'].append(Fe_transpose)
        if verbose:
            print(f"  Fe (transpose): {Fe_transpose:.6f}")
            sys.stdout.flush()

        # 4. CMA-ES optimized recovery
        seed = BEST_SEEDS.get(gamma, 42)
        results['best_seeds'].append(seed)

        if verbose:
            print(f"  Running CMA-ES (seed={seed})...")
            sys.stdout.flush()

        t0 = time.time()
        params, Fe_cmaes, info = optimize_cmaes_flat(
            logical_0, logical_1, gamma,
            N_depth=N_DEPTH, popsize=POPSIZE,
            maxiter=MAXITER, sigma0=SIGMA0,
            seed=seed, verbose=False,
        )
        elapsed = time.time() - t0

        results['Fe_cmaes'].append(float(Fe_cmaes))
        results['params'][f'gamma_{gamma}'] = np.array(params)

        if verbose:
            print(f"  Fe (CMA-ES):    {Fe_cmaes:.6f} ({elapsed:.0f}s)")
            sys.stdout.flush()

        # 5. Fock cross-validation
        alpha, beta = g(params, N_L)
        Fe_cmaes_coherent = float(entanglement_fidelity_displacement(
            alpha, beta,
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds, gamma))

        recovery_fock = channel_from_b(alpha, beta)
        Fe_cmaes_fock = float(entanglement_fidelity(
            recovery_fock, loss_ops, psi_0, psi_1))

        results['Fe_cmaes_fock'].append(Fe_cmaes_fock)
        gap = abs(Fe_cmaes_coherent - Fe_cmaes_fock)

        if verbose:
            print(f"  Fe (Fock):      {Fe_cmaes_fock:.6f} (gap={gap:.2e})")
            print(f"  Improvement:    +{Fe_cmaes - Fe_none:.4f} "
                  f"({100*(Fe_cmaes - Fe_none)/(1 - Fe_none):.1f}% of infidelity)")
            sys.stdout.flush()

    # Convert lists to arrays
    for key in ['Fe_none', 'Fe_sbs', 'Fe_cmaes', 'Fe_transpose',
                'Fe_cmaes_fock', 'best_seeds']:
        results[key] = np.array(results[key])

    return results


def save_results(results, filepath='results/cmaes_recovery_params.npz'):
    """Save benchmark results to NPZ file."""
    save_dict = {
        'gammas': results['gammas'],
        'Fe_none': results['Fe_none'],
        'Fe_sbs': results['Fe_sbs'],
        'Fe_cmaes': results['Fe_cmaes'],
        'Fe_transpose': results['Fe_transpose'],
        'Fe_cmaes_fock': results['Fe_cmaes_fock'],
        'best_seeds': results['best_seeds'],
        'N_depth': results['metadata']['N_depth'],
        'N_l': results['metadata']['N_l'],
        'Delta': results['metadata']['Delta'],
        'N_trunc': results['metadata']['N_trunc'],
        'popsize': results['metadata']['popsize'],
        'sigma0': results['metadata']['sigma0'],
    }

    # Add per-gamma parameters
    for key, params in results['params'].items():
        save_dict[f'params_{key}'] = params

    np.savez(filepath, **save_dict)
    print(f"\nSaved results to {filepath}")


def print_summary(results):
    """Print summary table."""
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'gamma':>8} | {'Fe(none)':>10} | {'Fe(SBS)':>10} | "
          f"{'Fe(CMA-ES)':>10} | {'Fe(tr)':>10} | {'Δ':>8}")
    print("-" * 70)

    for i, gamma in enumerate(results['gammas']):
        delta = results['Fe_cmaes'][i] - results['Fe_none'][i]
        print(f"{gamma:8.3f} | {results['Fe_none'][i]:10.6f} | "
              f"{results['Fe_sbs'][i]:10.6f} | {results['Fe_cmaes'][i]:10.6f} | "
              f"{results['Fe_transpose'][i]:10.6f} | {delta:+8.4f}")

    print("=" * 70)


if __name__ == "__main__":
    results = run_benchmark(verbose=True)
    print_summary(results)
    save_results(results)
