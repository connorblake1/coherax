"""
benchmark_claude.py

End-to-end benchmark comparing GKP recovery strategies under pure loss:
1. No recovery (identity channel after loss)
2. Transpose channel (theoretical upper bound)
3. Optimized CD+R circuit (main result)

Sweeps over gamma values and saves results for plotting.

Usage:
    python -m gkp_utils.benchmark_claude              # default settings
    python -m gkp_utils.benchmark_claude --quick       # fast test run
    python -m gkp_utils.benchmark_claude --full        # full benchmark
"""

import sys
import os
import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp

# Ensure the project root is on the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gkp_utils.transpose_channel_claude import (
    build_gkp_states,
    coherent_ket_to_fock,
    entanglement_fidelity,
    entanglement_fidelity_no_recovery,
    fock_transpose_recovery,
    GKP_N,
    dqdag,
)
from gkp_utils.recovery_optimizer_claude import (
    optimize_recovery,
    validate_in_fock,
)
from gkp_utils.characteristic_jax_utils import (
    make_pureloss_fock,
    make_transpose_for_pureloss,
)


def run_benchmark(
    gamma_values=None,
    Delta=0.3,
    N_trunc=3,
    T_depth=2,
    N_depth=6,
    lr=0.003,
    steps=5000,
    restarts=5,
    batch_size=32,
    loss_rank=10,
    save_dir="results_claude",
):
    """
    Run the full benchmark across gamma values.

    Args:
        gamma_values: list of loss parameters to test
        Delta: GKP envelope parameter
        N_trunc: coherent state truncation
        T_depth: traceout rounds (2^T_depth Kraus ops)
        N_depth: layers per round (2^N_depth displacement terms)
        lr: optimizer learning rate
        steps: gradient steps per restart
        restarts: random restarts
        batch_size: Bloch sphere samples per step
        loss_rank: number of loss Kraus operators
        save_dir: directory for saving results

    Returns:
        results: list of dicts with all fidelity data
    """
    if gamma_values is None:
        gamma_values = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1]

    print(f"JAX devices: {jax.devices()}")
    print(f"GKP: Delta={Delta}, N_trunc={N_trunc}, N_fock={GKP_N}")
    print(f"Circuit: T_depth={T_depth}, N_depth={N_depth}")
    print(f"Optimizer: lr={lr}, steps={steps}, restarts={restarts}, batch={batch_size}")
    print(f"Gamma values: {gamma_values}")
    print()

    # Build GKP states (shared across all gamma)
    logical_0, logical_1 = build_gkp_states(Delta=Delta, N_trunc=N_trunc)
    psi_0 = coherent_ket_to_fock(logical_0, GKP_N)
    psi_1 = coherent_ket_to_fock(logical_1, GKP_N)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(dqdag(psi_0) @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(dqdag(psi_1) @ psi_1).squeeze())

    N_l = 2 ** N_depth
    results = []

    for gamma in gamma_values:
        print(f"\n{'='*60}")
        print(f"  gamma = {gamma}")
        print(f"{'='*60}")
        t_start = time.time()

        # 1. No recovery baseline
        loss_ops = make_pureloss_fock(gamma, rank=loss_rank)
        Fe_none = float(entanglement_fidelity_no_recovery(loss_ops, psi_0, psi_1))
        print(f"  F_e (no recovery):  {Fe_none:.6f}")

        # 2. Transpose channel bound
        transpose_ops = make_transpose_for_pureloss(loss_ops, logical_0, logical_1)
        Fe_transpose = float(entanglement_fidelity(transpose_ops, loss_ops, psi_0, psi_1))
        print(f"  F_e (transpose):    {Fe_transpose:.6f}")

        # 3. Optimized CD+R circuit
        best_params, best_loss, loss_history = optimize_recovery(
            logical_0, logical_1, gamma,
            T_depth=T_depth, N_depth=N_depth,
            lr=lr, steps=steps, restarts=restarts,
            batch_size=batch_size, verbose=True,
        )

        Fe_analytic, Fe_fock = validate_in_fock(
            best_params, logical_0, logical_1, gamma, N_l, T_depth,
        )
        elapsed = time.time() - t_start

        print(f"\n  --- Results for gamma={gamma} ---")
        print(f"  F_e (no recovery):  {Fe_none:.6f}")
        print(f"  F_e (optimized):    {Fe_fock:.6f}  (analytic: {Fe_analytic:.6f})")
        print(f"  F_e (transpose):    {Fe_transpose:.6f}")
        print(f"  Gap to bound:       {Fe_transpose - Fe_fock:.6f}")

        result = {
            'gamma': gamma,
            'Fe_none': Fe_none,
            'Fe_optimized_analytic': Fe_analytic,
            'Fe_optimized_fock': Fe_fock,
            'Fe_transpose': Fe_transpose,
            'best_loss': float(best_loss),
            'best_params': np.array(best_params),
            'loss_history': loss_history,
            'elapsed_s': elapsed,
        }
        results.append(result)

        # Save intermediate results
        os.makedirs(save_dir, exist_ok=True)
        np.save(
            os.path.join(save_dir, f"result_gamma{gamma:.3f}.npy"),
            result, allow_pickle=True,
        )

    # Save full results
    np.save(os.path.join(save_dir, "all_results.npy"), results, allow_pickle=True)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Delta={Delta}, T_depth={T_depth}, N_depth={N_depth}")
    print(f"{'='*70}")
    print(f"{'gamma':>8s} | {'F_e(none)':>10s} | {'F_e(opt)':>10s} | {'F_e(trans)':>10s} | {'gap':>8s}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['gamma']:8.3f} | {r['Fe_none']:10.6f} | "
            f"{r['Fe_optimized_fock']:10.6f} | {r['Fe_transpose']:10.6f} | "
            f"{r['Fe_transpose']-r['Fe_optimized_fock']:8.6f}"
        )

    return results


def plot_results(results, save_path="results_claude/fidelity_comparison.png"):
    """Generate comparison plot of F_e vs gamma."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    gammas = [r['gamma'] for r in results]
    Fe_none = [r['Fe_none'] for r in results]
    Fe_opt = [r['Fe_optimized_fock'] for r in results]
    Fe_trans = [r['Fe_transpose'] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gammas, Fe_none, 'k--o', label='No recovery', markersize=5)
    ax.plot(gammas, Fe_opt, 'b-s', label='Optimized CD+R', markersize=6)
    ax.plot(gammas, Fe_trans, 'r-^', label='Transpose channel (bound)', markersize=6)

    ax.set_xlabel(r'Loss parameter $\gamma$')
    ax.set_ylabel(r'Entanglement fidelity $F_e$')
    ax.set_title(r'GKP Loss Recovery ($\Delta=0.3$, square lattice)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="GKP loss recovery benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with small circuit and few steps")
    parser.add_argument("--full", action="store_true",
                        help="Full benchmark with large circuit")
    parser.add_argument("--gamma", type=float, nargs="+", default=None,
                        help="Specific gamma values to test")
    args = parser.parse_args()

    if args.quick:
        results = run_benchmark(
            gamma_values=args.gamma or [0.03, 0.05],
            T_depth=1, N_depth=4,
            lr=0.005, steps=1000, restarts=2, batch_size=8,
            save_dir="results_claude_quick",
        )
    elif args.full:
        results = run_benchmark(
            gamma_values=args.gamma or [0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
            T_depth=2, N_depth=6,
            lr=0.003, steps=10000, restarts=10, batch_size=64,
            save_dir="results_claude_full",
        )
    else:
        results = run_benchmark(
            gamma_values=args.gamma or [0.01, 0.03, 0.05, 0.07, 0.1],
            T_depth=2, N_depth=6,
            lr=0.003, steps=5000, restarts=5, batch_size=32,
            save_dir="results_claude",
        )

    plot_results(results)


if __name__ == "__main__":
    main()
