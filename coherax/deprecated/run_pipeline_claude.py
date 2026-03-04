"""
run_pipeline_claude.py

Driver script for systematic CD+R recovery optimization using CMA-ES.

Runs BIPOP-CMA-ES across gamma values and N_depth, collects results,
performs Fock cross-validation, and saves a summary.

Key findings informing this pipeline:
  - CMA-ES with N_depth=6 is the validated sweet spot for flat recovery
  - The landscape is highly multimodal: ~10-20% of seeds find improvement
  - Gradient descent ALWAYS converges to identity; only CMA-ES works
  - Fock cross-validation perfectly matches analytic Fe (gap < 1e-6)

Usage:
    python -m coherax.run_pipeline_claude [--gammas 0.03,0.05,0.07,0.1]
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys
import time
import json
from pathlib import Path

from coherax.characteristic_jax_utils import (
    gkp_coherent_dm, g, GKP_N, dqcoherent, make_pureloss_fock,
    apply_kraus_map_nonorm, channel_from_b,
)
from coherax.coherent_tree_optimizer_claude import (
    entanglement_fidelity_displacement,
    bipop_cmaes_flat,
    hybrid_cmaes_gradient,
)


def fock_cross_validate(params, N_depth, logical_0, logical_1, gamma,
                         loss_rank=10, N=GKP_N):
    """
    Cross-validate analytic Fe with Fock-basis computation.

    Returns:
        Fe_fock: entanglement fidelity in Fock basis
    """
    N_l = 2 ** N_depth
    alpha, beta = g(params, N_l)

    recovery_ops = channel_from_b(alpha, beta)
    loss_ops = make_pureloss_fock(gamma, rank=loss_rank, N=N)

    fock_states = []
    for ck in [logical_0, logical_1]:
        coherents = jnp.squeeze(
            jax.vmap(lambda a: dqcoherent(N, a))(ck.ds))
        if coherents.ndim == 3:
            coherents = coherents.squeeze(-1)
        psi = jnp.einsum('bn,b->n', coherents, ck.cs).reshape(-1, 1)
        psi = psi / jnp.sqrt(jnp.real(jnp.conj(psi).T @ psi).squeeze())
        fock_states.append(psi)

    Fe = 0.0
    for mu in range(2):
        for nu in range(2):
            rho_mn = fock_states[mu] @ jnp.conj(fock_states[nu]).T
            after_loss = apply_kraus_map_nonorm(loss_ops, rho_mn)
            after_recovery = apply_kraus_map_nonorm(recovery_ops, after_loss)
            Fe += (jnp.conj(fock_states[mu]).T @ after_recovery
                   @ fock_states[nu]).squeeze()
    return float(jnp.real(Fe) / 4.0)


def run_sweep(
    gammas=(0.03, 0.05, 0.07, 0.1, 0.15),
    N_depth=6,
    n_restarts=20,
    popsize=80,
    maxiter=1000,
    Delta=0.3,
    N_trunc=3,
    lattice='square',
    do_fock_validation=True,
    verbose=True,
):
    """
    Run systematic CMA-ES optimization sweep across gamma values.

    Args:
        gammas: tuple of gamma values to sweep
        N_depth: CD+R circuit depth
        n_restarts: CMA-ES restarts per gamma
        popsize: CMA-ES population size
        maxiter: max generations per restart
        Delta, N_trunc, lattice: GKP state parameters
        do_fock_validation: whether to cross-validate in Fock basis
        verbose: print progress

    Returns:
        results: dict mapping gamma -> result dict
    """
    results = {}

    for gamma in gammas:
        if verbose:
            print(f"\n{'='*60}")
            print(f"gamma={gamma}")
            print(f"{'='*60}")
            sys.stdout.flush()

        logical_0 = gkp_coherent_dm(
            mu=0, N_trunc=N_trunc, Delta=Delta, lattice=lattice)
        logical_1 = gkp_coherent_dm(
            mu=1, N_trunc=N_trunc, Delta=Delta, lattice=lattice)

        # Compute baseline
        Fe_id = float(entanglement_fidelity_displacement(
            jnp.ones((1, 1), dtype=jnp.complex64),
            jnp.zeros((1, 1), dtype=jnp.complex64),
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds, gamma))

        if verbose:
            print(f"Fe_id = {Fe_id:.6f}")
            sys.stdout.flush()

        # BIPOP CMA-ES
        best_params, best_Fe, info = bipop_cmaes_flat(
            logical_0, logical_1, gamma,
            N_depth=N_depth, n_restarts=n_restarts,
            popsize=popsize, maxiter=maxiter,
            verbose=verbose,
        )

        result = {
            'gamma': gamma,
            'Fe_id': Fe_id,
            'Fe_cma': best_Fe,
            'improvement': best_Fe - Fe_id,
            'n_improved': info['n_improved'],
            'n_restarts': n_restarts,
            'total_time': info['total_time'],
            'N_depth': N_depth,
        }

        # Fock cross-validation on best result
        if do_fock_validation and best_Fe > Fe_id + 0.001:
            if verbose:
                print("  Fock cross-validation...")
                sys.stdout.flush()
            Fe_fock = fock_cross_validate(
                best_params, N_depth, logical_0, logical_1, gamma)
            result['Fe_fock'] = Fe_fock
            result['fock_gap'] = abs(best_Fe - Fe_fock)
            if verbose:
                print(f"  Fe_fock={Fe_fock:.6f}, "
                      f"gap={abs(best_Fe - Fe_fock):.6f}")
                sys.stdout.flush()

        # Save best circuit params as list for JSON serialization
        result['best_params'] = {
            f'layer_{i}': {
                'd_re': float(best_params[i, 0].real),
                'd_im': float(best_params[i, 0].imag),
                'd_abs': float(jnp.abs(best_params[i, 0])),
                'phi': float(best_params[i, 1].real),
                'theta': float(best_params[i, 2].real),
            }
            for i in range(N_depth)
        }

        results[gamma] = result

    return results


def print_summary(results):
    """Print a formatted summary table of results."""
    print(f"\n{'='*70}")
    print(f"CMA-ES Recovery Optimization Summary (N_depth=6, Delta=0.3)")
    print(f"{'='*70}")
    print(f"{'gamma':>8s} {'Fe_id':>10s} {'Fe_cma':>10s} "
          f"{'Fe_fock':>10s} {'improve':>10s} {'hits':>8s} {'time':>8s}")
    print(f"{'-'*70}")

    for gamma in sorted(results.keys()):
        r = results[gamma]
        fe_fock = f"{r.get('Fe_fock', 0):10.6f}" if 'Fe_fock' in r else "      n/a "
        improve = r['improvement']
        flag = ' *' if improve > 0.001 else ''
        print(f"{gamma:8.3f} {r['Fe_id']:10.6f} {r['Fe_cma']:10.6f} "
              f"{fe_fock} {improve:+10.6f} "
              f"{r['n_improved']:3d}/{r['n_restarts']:3d} "
              f"{r['total_time']:6.0f}s{flag}")

    print(f"{'='*70}")
    sys.stdout.flush()


def save_results(results, filename="cma_es_results.json"):
    """Save results to JSON file."""
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    filepath = output_dir / filename

    # Convert to JSON-serializable format
    json_results = {}
    for gamma, r in results.items():
        json_results[str(gamma)] = {
            k: v for k, v in r.items()
            if k != 'best_params_jax'
        }

    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"Results saved to {filepath}")
    return filepath


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='CMA-ES recovery optimization sweep')
    parser.add_argument('--gammas', type=str,
                        default='0.03,0.05,0.07,0.1,0.15',
                        help='Comma-separated gamma values')
    parser.add_argument('--N_depth', type=int, default=6,
                        help='CD+R circuit depth')
    parser.add_argument('--restarts', type=int, default=20,
                        help='CMA-ES restarts per gamma')
    parser.add_argument('--popsize', type=int, default=80,
                        help='CMA-ES population size')
    parser.add_argument('--maxiter', type=int, default=1000,
                        help='Max CMA-ES generations')
    parser.add_argument('--no-fock', action='store_true',
                        help='Skip Fock cross-validation')
    args = parser.parse_args()

    gammas = tuple(float(g) for g in args.gammas.split(','))

    print("=" * 70)
    print("CD+R Recovery Optimization Pipeline")
    print(f"  gammas: {gammas}")
    print(f"  N_depth: {args.N_depth}, restarts: {args.restarts}")
    print(f"  popsize: {args.popsize}, maxiter: {args.maxiter}")
    print("=" * 70)
    sys.stdout.flush()

    results = run_sweep(
        gammas=gammas,
        N_depth=args.N_depth,
        n_restarts=args.restarts,
        popsize=args.popsize,
        maxiter=args.maxiter,
        do_fock_validation=not args.no_fock,
    )

    print_summary(results)
    save_results(results)
