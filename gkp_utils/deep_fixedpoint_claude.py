"""
deep_fixedpoint_claude.py

Test deeper circuits (N_depth = 6, 8, 10, 12) for code-space fixed-point optimization.
Saves all best parameters and results to NPZ file.
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys
import time
from typing import Dict, Any, Tuple

from gkp_utils.characteristic_jax_utils import (
    CoherentKet,
    g,
    gkp_coherent_dm,
    make_pureloss_fock,
    apply_kraus_map_nonorm,
    dqcoherent,
    dag,
    GKP_N,
    dqdisplace,
)

from gkp_utils.fixedpoint_optimizer_claude import (
    channel_from_b_N,
    compute_code_space_channel,
    code_space_fixed_point_loss,
    bloch_sphere_fidelities,
    multiround_fidelity_fock,
)


def deep_fixedpoint_cmaes(
    logical_0: CoherentKet,
    logical_1: CoherentKet,
    gamma: float,
    N_depth: int,
    popsize: int = 40,
    maxiter: int = 500,
    sigma0: float = 3.0,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[jnp.ndarray, float, Dict[str, Any]]:
    """
    CMA-ES optimization for code-space fixed point with arbitrary depth.

    Returns:
        best_params: (N_depth, 4) optimized parameters
        best_loss: ||S - I||_F achieved
        info: dict with diagnostics
    """
    import cma

    N_l = 2 ** N_depth
    n_params = N_depth * 4
    d_half = float(jnp.sqrt(jnp.pi / 2))

    # Fock basis dimensions
    N = min(GKP_N, 80)
    loss_rank = min(10, N // 2)

    def unpack(x_real):
        p = jnp.zeros((N_depth, 4), dtype=jnp.complex64)
        for i in range(N_depth):
            p = p.at[i, 0].set(x_real[4*i] + 1j * x_real[4*i+1])
            p = p.at[i, 1].set(x_real[4*i+2])
            p = p.at[i, 2].set(x_real[4*i+3])
        return p

    # Build Fock operators
    loss_ops = make_pureloss_fock(gamma, rank=loss_rank, N=N)

    fock_states = []
    for ck in [logical_0, logical_1]:
        coherents = jnp.squeeze(jax.vmap(lambda a: dqcoherent(N, a))(ck.ds))
        if coherents.ndim == 3:
            coherents = coherents.squeeze(-1)
        psi = jnp.einsum('bn,b->n', coherents, ck.cs).reshape(-1, 1)
        psi = psi / jnp.sqrt(jnp.real(dag(psi) @ psi).squeeze())
        fock_states.append(psi)

    psi_0, psi_1 = fock_states

    # JIT-compile the core computation
    @jax.jit
    def compute_loss_jit(params_flat):
        p = jnp.zeros((N_depth, 4), dtype=jnp.complex64)
        for i in range(N_depth):
            p = p.at[i, 0].set(params_flat[4*i] + 1j * params_flat[4*i+1])
            p = p.at[i, 1].set(params_flat[4*i+2])
            p = p.at[i, 2].set(params_flat[4*i+3])
        alpha, beta = g(p, N_l)
        recovery_ops = channel_from_b_N(alpha, beta, N)
        S = compute_code_space_channel(recovery_ops, loss_ops, psi_0, psi_1)
        return code_space_fixed_point_loss(S)

    def objective_fn(x):
        return float(compute_loss_jit(jnp.array(x, dtype=jnp.float32)))

    # Warm-up JIT
    print(f"    JIT compiling... ", end="", flush=True)
    _ = objective_fn(np.zeros(n_params))
    print("done", flush=True)

    # Initial point with alternating CD pattern
    x0 = np.zeros(n_params)
    for i in range(N_depth):
        if i % 2 == 0:
            x0[4*i] = d_half
            x0[4*i + 3] = np.pi/2
        else:
            x0[4*i] = d_half
            x0[4*i + 3] = np.pi/2

    baseline = objective_fn(np.zeros(n_params))

    if verbose:
        print(f"  N_depth={N_depth}, N_l={N_l}, params={n_params}")
        print(f"  Baseline ||S-I||_F = {baseline:.6f}")
        sys.stdout.flush()

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'maxiter': maxiter,
        'popsize': popsize,
        'verbose': -1,
        'seed': seed,
        'tolfun': 1e-11,
    })

    gen = 0
    best_ever = baseline
    t_start = time.time()

    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective_fn(x) for x in solutions]
        es.tell(solutions, fitnesses)

        best_now = es.result.fbest
        if best_now < best_ever:
            best_ever = best_now

        if verbose and gen % 50 == 0:
            elapsed = time.time() - t_start
            print(f"    gen {gen}: ||S-I||_F = {best_now:.6f} (ever={best_ever:.6f}) [{elapsed:.0f}s]", flush=True)
        gen += 1

    best_params = unpack(es.result.xbest)
    best_loss = es.result.fbest
    elapsed = time.time() - t_start

    # Final diagnostics
    alpha, beta = g(best_params, N_l)
    recovery_ops = channel_from_b_N(alpha, beta, N)
    S_final = compute_code_space_channel(recovery_ops, loss_ops, psi_0, psi_1)
    fids_final = bloch_sphere_fidelities(recovery_ops, loss_ops, psi_0, psi_1, n_points=6)

    # Multi-round fidelities
    round_fes = []
    for r in [1, 2, 3, 5, 10, 20]:
        fe = multiround_fidelity_fock(best_params, logical_0, logical_1, gamma, n_rounds=r, N=N)
        round_fes.append((r, fe))

    if verbose:
        print(f"    Done ({elapsed:.0f}s, {gen} gens)")
        print(f"    Final ||S-I||_F = {best_loss:.6f}")
        print(f"    Bloch fidelities: min={float(jnp.min(fids_final)):.4f}, "
              f"max={float(jnp.max(fids_final)):.4f}")
        print(f"    Round fidelities: ", end="")
        for r, fe in round_fes[:4]:
            print(f"R{r}={fe:.4f} ", end="")
        print()
        sys.stdout.flush()

    return best_params, best_loss, {
        'S_final': S_final,
        'bloch_fidelities': fids_final,
        'round_fidelities': round_fes,
        'generations': gen,
        'elapsed': elapsed,
        'baseline': baseline,
    }


def run_depth_comparison(
    gamma: float = 0.05,
    depths: list = [6, 8, 10, 12],
    n_seeds: int = 3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run fixed-point optimization for multiple depths.

    For each depth, runs multiple seeds and keeps best result.
    """

    print("=" * 70)
    print("Deep Circuit Fixed-Point Optimizer")
    print("=" * 70)
    print(f"gamma={gamma}, depths={depths}, seeds_per_depth={n_seeds}")
    print(f"GKP: Delta=0.3, N_trunc=3")
    print("=" * 70)
    sys.stdout.flush()

    # Build GKP states
    logical_0 = gkp_coherent_dm(mu=0, N_trunc=3, Delta=0.3, lattice='square')
    logical_1 = gkp_coherent_dm(mu=1, N_trunc=3, Delta=0.3, lattice='square')

    results = {}

    for N_depth in depths:
        print(f"\n{'='*60}")
        print(f"N_depth = {N_depth}")
        print(f"{'='*60}")
        sys.stdout.flush()

        best_params = None
        best_loss = float('inf')
        best_info = None
        best_seed = None

        for seed in range(n_seeds):
            print(f"\n--- Seed {seed} ---")
            sys.stdout.flush()

            params, loss, info = deep_fixedpoint_cmaes(
                logical_0, logical_1, gamma,
                N_depth=N_depth,
                popsize=40,
                maxiter=500,
                sigma0=3.0,
                seed=seed,
                verbose=verbose,
            )

            if loss < best_loss:
                best_loss = loss
                best_params = params
                best_info = info
                best_seed = seed
                print(f"  >> New best! ||S-I||_F = {best_loss:.6f}")

        results[N_depth] = {
            'params': best_params,
            'loss': best_loss,
            'info': best_info,
            'best_seed': best_seed,
        }

        print(f"\nBest for N_depth={N_depth}: ||S-I||_F = {best_loss:.6f} (seed={best_seed})")

    return results, logical_0, logical_1


def save_results(
    results: Dict[int, Dict],
    logical_0: CoherentKet,
    logical_1: CoherentKet,
    gamma: float,
    filename: str = 'results/deep_fixedpoint_results.npz',
):
    """Save all results to NPZ file."""

    data = {
        'gamma': gamma,
        'Delta': 0.3,
        'N_trunc': 3,
        'depths': np.array(list(results.keys())),
    }

    for depth, res in results.items():
        prefix = f'd{depth}_'
        data[prefix + 'params'] = np.array(res['params'])
        data[prefix + 'loss'] = res['loss']
        data[prefix + 'baseline'] = res['info']['baseline']
        data[prefix + 'bloch_fidelities'] = np.array(res['info']['bloch_fidelities'])
        data[prefix + 'round_fidelities'] = np.array(res['info']['round_fidelities'])
        data[prefix + 'S_final'] = np.array(res['info']['S_final'])
        data[prefix + 'best_seed'] = res['best_seed']
        data[prefix + 'elapsed'] = res['info']['elapsed']

    np.savez(filename, **data)
    print(f"\nResults saved to {filename}")


def print_summary(results: Dict[int, Dict]):
    """Print summary table."""

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Depth':<8} {'||S-I||_F':<12} {'Min Bloch':<12} {'Fe@R1':<10} "
          f"{'Fe@R5':<10} {'Fe@R10':<10} {'Fe@R20':<10}")
    print("-" * 70)

    for depth in sorted(results.keys()):
        res = results[depth]
        fids = res['info']['bloch_fidelities']
        rounds = dict(res['info']['round_fidelities'])

        print(f"{depth:<8} {res['loss']:<12.6f} {float(jnp.min(fids)):<12.4f} "
              f"{rounds.get(1, 0):<10.4f} {rounds.get(5, 0):<10.4f} "
              f"{rounds.get(10, 0):<10.4f} {rounds.get(20, 0):<10.4f}")

    print("-" * 70)

    # Find best depth
    best_depth = min(results.keys(), key=lambda d: results[d]['loss'])
    print(f"\nBest: N_depth={best_depth} with ||S-I||_F = {results[best_depth]['loss']:.6f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.05)
    parser.add_argument('--depths', nargs='+', type=int, default=[6, 8, 10, 12])
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--output', type=str, default='results/deep_fixedpoint_results.npz')
    args = parser.parse_args()

    results, log0, log1 = run_depth_comparison(
        gamma=args.gamma,
        depths=args.depths,
        n_seeds=args.seeds,
        verbose=True,
    )

    print_summary(results)
    save_results(results, log0, log1, args.gamma, args.output)
