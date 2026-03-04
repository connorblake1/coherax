"""
benchmark_low_gamma_claude.py

Benchmark for low gamma values (0.001, 0.01) to verify limit cases.
Uses BIPOP multi-restart CMA-ES to find good seeds.

Usage:
    python -m gkp_utils.benchmark_low_gamma_claude
"""

import jax
import jax.numpy as jnp
import numpy as np
import json
import time
import sys
import os

from gkp_utils.characteristic_jax_utils import (
    g, channel_from_b, make_pureloss_fock, make_transpose_for_pureloss,
    GKP_N,
)
from gkp_utils.transpose_channel_claude import (
    build_gkp_states, coherent_ket_to_fock,
    build_sbs_kraus, entanglement_fidelity,
    entanglement_fidelity_no_recovery,
)
from gkp_utils.coherent_tree_optimizer_claude import (
    entanglement_fidelity_displacement,
    bipop_cmaes_flat,
)


# Configuration
GAMMAS = [0.001, 0.01]
N_DEPTH = 6
N_L = 2 ** N_DEPTH
DELTA = 0.3
N_TRUNC = 3
LOSS_RANK = 10
N_RESTARTS = 10  # BIPOP restarts to find good seeds


def fock_cross_validate(params, logical_0, logical_1, gamma,
                        N_depth=N_DEPTH, loss_rank=LOSS_RANK, N=GKP_N):
    """Cross-validate by computing Fe in Fock basis."""
    N_l = 2 ** N_depth
    alpha, beta = g(params, N_l)
    recovery_ops = channel_from_b(alpha, beta)

    psi_0 = coherent_ket_to_fock(logical_0, N)
    psi_1 = coherent_ket_to_fock(logical_1, N)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(jnp.conj(psi_0).T @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(jnp.conj(psi_1).T @ psi_1).squeeze())

    loss_ops = make_pureloss_fock(gamma, rank=loss_rank, N=N)
    Fe_fock = entanglement_fidelity(recovery_ops, loss_ops, psi_0, psi_1)
    return float(Fe_fock)


def run_low_gamma_benchmark(save_dir="results"):
    """Run benchmark for low gamma values."""
    print("=" * 70)
    print("  Low Gamma Benchmark (0.001, 0.01)")
    print("=" * 70)
    print(f"  N_depth={N_DEPTH}, N_l={N_L}, Delta={DELTA}")
    print(f"  BIPOP restarts={N_RESTARTS}")
    print()

    # Build GKP states
    print("Building GKP states...", end=" ")
    sys.stdout.flush()
    logical_0, logical_1 = build_gkp_states(
        Delta=DELTA, N_trunc=N_TRUNC, lattice="square")
    psi_0 = coherent_ket_to_fock(logical_0)
    psi_1 = coherent_ket_to_fock(logical_1)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(jnp.conj(psi_0).T @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(jnp.conj(psi_1).T @ psi_1).squeeze())
    print("done.")

    # Build SBS
    print("Building SBS Kraus operators...", end=" ")
    sys.stdout.flush()
    sbs_kraus = build_sbs_kraus(Delta=DELTA)
    print("done.")

    results = {}

    for gamma in GAMMAS:
        print(f"\n{'=' * 60}")
        print(f"  gamma = {gamma}")
        print(f"{'=' * 60}")
        t0 = time.time()

        # Loss operators
        loss_ops = make_pureloss_fock(gamma, rank=LOSS_RANK)

        # Identity
        Fe_none = float(entanglement_fidelity_no_recovery(
            loss_ops, psi_0, psi_1))
        print(f"  Fe_none (identity)   = {Fe_none:.6f}")

        # SBS
        Fe_sbs = float(entanglement_fidelity(
            sbs_kraus, loss_ops, psi_0, psi_1))
        print(f"  Fe_sbs               = {Fe_sbs:.6f}")

        # Transpose channel
        transpose_ops = make_transpose_for_pureloss(
            loss_ops, logical_0, logical_1)
        Fe_transpose = float(entanglement_fidelity(
            transpose_ops, loss_ops, psi_0, psi_1))
        print(f"  Fe_transpose         = {Fe_transpose:.6f}")

        # BIPOP CMA-ES
        print(f"\n  Running BIPOP CMA-ES ({N_RESTARTS} restarts)...")
        sys.stdout.flush()

        params, Fe_cmaes, info = bipop_cmaes_flat(
            logical_0, logical_1, gamma,
            N_depth=N_DEPTH, n_restarts=N_RESTARTS,
            popsize=80, maxiter=1500, sigma0=3.0,
            verbose=True,
        )
        print(f"  Fe_cmaes (coherent)  = {Fe_cmaes:.6f}")
        print(f"  Best seed: {info['trials'][np.argmax([t['Fe'] for t in info['trials']])]['seed']}")

        # Fock cross-validation
        Fe_fock = fock_cross_validate(params, logical_0, logical_1, gamma)
        gap = abs(Fe_cmaes - Fe_fock)
        print(f"  Fe_cmaes (Fock)      = {Fe_fock:.6f}  (gap = {gap:.2e})")

        elapsed = time.time() - t0
        print(f"\n  Time: {elapsed:.0f}s")

        results[gamma] = {
            "gamma": gamma,
            "Fe_none": Fe_none,
            "Fe_sbs": Fe_sbs,
            "Fe_cmaes": Fe_cmaes,
            "Fe_cmaes_fock": Fe_fock,
            "Fe_transpose": Fe_transpose,
            "best_seed": info['trials'][np.argmax([t['Fe'] for t in info['trials']])]['seed'],
            "params": np.array(params),
            "improvement": Fe_cmaes - Fe_none,
            "pct_infidelity_recovered": 100 * (Fe_cmaes - Fe_none) / (1 - Fe_none) if Fe_none < 1 else 0,
        }

    # Summary table
    print(f"\n\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'gamma':>8s} | {'Fe_none':>10s} | {'Fe_CMA-ES':>10s} | "
          f"{'Fe_transp':>10s} | {'improv':>10s} | {'% recov':>8s}")
    print("-" * 70)
    for gamma in GAMMAS:
        r = results[gamma]
        print(f"{gamma:8.4f} | {r['Fe_none']:10.6f} | {r['Fe_cmaes']:10.6f} | "
              f"{r['Fe_transpose']:10.6f} | {r['improvement']:+10.6f} | "
              f"{r['pct_infidelity_recovered']:7.1f}%")

    # Save results
    os.makedirs(save_dir, exist_ok=True)

    # Save to NPZ
    save_dict = {
        "gammas": np.array(GAMMAS),
        "N_depth": N_DEPTH,
        "Delta": DELTA,
    }
    for gamma in GAMMAS:
        g_key = f"{gamma:.4f}".replace(".", "p")
        save_dict[f"Fe_none_{g_key}"] = results[gamma]["Fe_none"]
        save_dict[f"Fe_sbs_{g_key}"] = results[gamma]["Fe_sbs"]
        save_dict[f"Fe_cmaes_{g_key}"] = results[gamma]["Fe_cmaes"]
        save_dict[f"Fe_transpose_{g_key}"] = results[gamma]["Fe_transpose"]
        save_dict[f"params_{g_key}"] = results[gamma]["params"]
        save_dict[f"best_seed_{g_key}"] = results[gamma]["best_seed"]

    npz_path = os.path.join(save_dir, "low_gamma_results.npz")
    np.savez(npz_path, **save_dict)
    print(f"\n  Results saved to: {npz_path}")

    # Save JSON summary
    json_results = {str(g): {k: v for k, v in r.items() if k != "params"}
                    for g, r in results.items()}
    json_path = os.path.join(save_dir, "low_gamma_summary.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"  Summary saved to: {json_path}")

    return results


if __name__ == "__main__":
    run_low_gamma_benchmark()
