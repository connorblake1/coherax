"""
multi_gamma_sweep_claude.py

Run depth sweep for multiple gamma values: 0.03 and 0.1
Uses the optimal warm-start chain: 7 → 8 → 10

Usage:
    python -m coherax.multi_gamma_sweep_claude
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
DELTA = 0.3
N_TRUNC = 3
POPSIZE = 120
MAXITER = 1500
SIGMA0 = 2.5
DEPTHS = [7, 8, 10]  # Optimal chain


def sbs_informed_init(N_depth):
    """Initialize parameters inspired by SBS recovery."""
    d_half = float(jnp.sqrt(jnp.pi / 2))
    params = np.zeros((N_depth, 4), dtype=np.complex64)
    for i in range(N_depth):
        if i % 2 == 0:
            params[i, 0] = d_half * (1 + 0.1j * (i//2))
            params[i, 2] = np.pi / 2
        else:
            params[i, 0] = d_half * (0.1 + 1j)
            params[i, 2] = 0
    return jnp.array(params, dtype=jnp.complex64)


def extend_params(best_params, N_depth_new):
    """Extend parameters from previous depth to new depth."""
    N_depth_old = best_params.shape[0]
    if N_depth_new <= N_depth_old:
        return best_params[:N_depth_new]

    params_new = jnp.zeros((N_depth_new, 4), dtype=jnp.complex64)
    params_new = params_new.at[:N_depth_old].set(best_params)

    for i in range(N_depth_old, N_depth_new):
        src_idx = (i - N_depth_old) % N_depth_old
        scale = 0.3 / (i - N_depth_old + 1)
        params_new = params_new.at[i, 0].set(best_params[src_idx, 0] * scale)
        params_new = params_new.at[i, 1].set(best_params[src_idx, 1])
        params_new = params_new.at[i, 2].set(best_params[src_idx, 2] + 0.05)

    return params_new


def optimize_cmaes(
    logical_0, logical_1, gamma,
    N_depth, init_params=None, init_type="cold",
    popsize=120, maxiter=1500, sigma0=2.5,
    seed=42, verbose=True,
):
    """CMA-ES optimization."""
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

    if init_params is not None:
        x0 = pack(init_params)
        sigma = sigma0 * 0.5  # Smaller for warm-start
    else:
        x0 = pack(sbs_informed_init(N_depth))
        sigma = sigma0

    Fe_id = float(eval_circuit(jnp.zeros((N_depth, 4), dtype=jnp.complex64)))
    Fe_init = -objective(x0)

    if verbose:
        print(f"  N_depth={N_depth}, N_l={N_l}, params={n_params}")
        print(f"  Init: {init_type}, Fe_init={Fe_init:.6f}, Fe_id={Fe_id:.6f}")
        sys.stdout.flush()

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

    return best_params, best_fe, Fe_id


def run_gamma_sweep(gamma, depths=DEPTHS, verbose=True):
    """Run depth sweep for a single gamma value."""
    # Build GKP states
    logical_0, logical_1 = build_gkp_states(Delta=DELTA, N_trunc=N_TRUNC)
    psi_0 = coherent_ket_to_fock(logical_0, GKP_N)
    psi_1 = coherent_ket_to_fock(logical_1, GKP_N)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(dqdag(psi_0) @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(dqdag(psi_1) @ psi_1).squeeze())

    # Baseline
    loss_ops = make_pureloss_fock(gamma, rank=10, N=GKP_N)
    Fe_none = float(entanglement_fidelity_no_recovery(loss_ops, psi_0, psi_1))

    if verbose:
        print("=" * 70)
        print(f"GAMMA = {gamma}")
        print(f"Fe(none) = {Fe_none:.6f}")
        print("=" * 70)
        sys.stdout.flush()

    results = {
        'gamma': gamma,
        'Fe_none': Fe_none,
        'depths': [],
        'Fe_best': [],
        'params_best': {},
    }

    current_best_params = None
    current_best_fe = 0
    current_best_depth = None

    for depth in depths:
        if verbose:
            print(f"\n{'='*60}")
            print(f"N_depth = {depth} (N_l = {2**depth}, params = {4*depth})")
            print(f"{'='*60}")
            sys.stdout.flush()

        best_fe = 0
        best_params = None

        # Strategy 1: Cold-start with multiple seeds (for depth 7)
        if current_best_params is None:
            for seed in range(3):
                if verbose:
                    print(f"\n--- Cold-start seed {seed} ---")
                    sys.stdout.flush()

                params, fe, fe_id = optimize_cmaes(
                    logical_0, logical_1, gamma,
                    N_depth=depth, init_params=None, init_type=f"cold-seed{seed}",
                    popsize=POPSIZE, maxiter=MAXITER, sigma0=SIGMA0,
                    seed=seed, verbose=verbose,
                )

                if fe > best_fe:
                    best_fe = fe
                    best_params = params
        else:
            # Strategy 2: Warm-start from previous best
            if verbose:
                print(f"\n--- Warm-start from depth {current_best_depth} (Fe={current_best_fe:.6f}) ---")
                sys.stdout.flush()

            init_p = extend_params(current_best_params, depth)
            params, fe, fe_id = optimize_cmaes(
                logical_0, logical_1, gamma,
                N_depth=depth, init_params=init_p,
                init_type=f"warm-from-depth{current_best_depth}",
                popsize=POPSIZE, maxiter=MAXITER, sigma0=SIGMA0,
                seed=42, verbose=verbose,
            )
            best_fe = fe
            best_params = params

        results['depths'].append(depth)
        results['Fe_best'].append(best_fe)
        results['params_best'][depth] = np.array(best_params)

        # Update best for warm-starting
        if best_fe > current_best_fe:
            current_best_fe = best_fe
            current_best_params = best_params
            current_best_depth = depth

        if verbose:
            delta = best_fe - Fe_none
            pct = 100 * delta / (1.0 - Fe_none)
            print(f"\n  Summary depth {depth}: Fe={best_fe:.6f}, +{delta:.4f} ({pct:.1f}% recovered)")
            sys.stdout.flush()

    return results


def run_multi_gamma(gammas=[0.03, 0.1], verbose=True):
    """Run depth sweep for multiple gamma values."""
    all_results = {}

    for gamma in gammas:
        results = run_gamma_sweep(gamma, depths=DEPTHS, verbose=verbose)
        all_results[gamma] = results

    # Print summary
    print("\n" + "=" * 80)
    print("MULTI-GAMMA SUMMARY")
    print("=" * 80)
    print(f"{'Gamma':>8} | {'Fe(none)':>10} | {'Fe(d=7)':>10} | {'Fe(d=8)':>10} | {'Fe(d=10)':>10} | {'Best':>10} | {'% Rec':>8}")
    print("-" * 80)

    for gamma, res in all_results.items():
        Fe_none = res['Fe_none']
        Fe_7 = res['Fe_best'][0] if len(res['Fe_best']) > 0 else 0
        Fe_8 = res['Fe_best'][1] if len(res['Fe_best']) > 1 else 0
        Fe_10 = res['Fe_best'][2] if len(res['Fe_best']) > 2 else 0
        best = max(res['Fe_best'])
        pct = 100 * (best - Fe_none) / (1.0 - Fe_none)
        print(f"{gamma:8.2f} | {Fe_none:10.6f} | {Fe_7:10.6f} | {Fe_8:10.6f} | {Fe_10:10.6f} | {best:10.6f} | {pct:7.1f}%")

    print("=" * 80)

    # Save results
    save_dict = {}
    for gamma, res in all_results.items():
        prefix = f"gamma_{gamma:.2f}".replace(".", "p")
        save_dict[f'{prefix}_Fe_none'] = res['Fe_none']
        save_dict[f'{prefix}_depths'] = np.array(res['depths'])
        save_dict[f'{prefix}_Fe_best'] = np.array(res['Fe_best'])
        for depth, params in res['params_best'].items():
            save_dict[f'{prefix}_params_depth_{depth}'] = params

    np.savez('results/multi_gamma_sweep.npz', **save_dict)
    print("\nSaved to results/multi_gamma_sweep.npz")

    return all_results


if __name__ == "__main__":
    run_multi_gamma(gammas=[0.03, 0.1], verbose=True)
