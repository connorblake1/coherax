"""
multi_round_claude.py

Multi-round continuous stabilization simulation.

Simulates repeated cycles of (pure loss → recovery) and tracks
entanglement fidelity Fe over N rounds. Compares four strategies:
  1. Identity (no recovery)
  2. SBS (Small-Big-Small stabilizer measurement)
  3. CMA-ES optimized CD+R circuits
  4. Transpose channel (theoretical optimal among transpose-type maps)

All computation runs in the Fock basis.

Usage:
    python -m coherax.multi_round_claude
    python -m coherax.multi_round_claude --rounds 30 --gammas 0.03 0.05
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import sys
import os
import argparse

from coherax.characteristic_jax_utils import (
    g, channel_from_b, make_pureloss_fock, make_transpose_for_pureloss,
    apply_kraus_map_nonorm, dag, GKP_N, dqtrace,
)
from coherax.transpose_channel_claude import (
    build_gkp_states, coherent_ket_to_fock, build_sbs_kraus,
)


# ============================================================
# CORE SIMULATION
# ============================================================

def compute_fe_from_rhos(rhos, psi_0, psi_1):
    """
    Compute entanglement fidelity from the 4 tracked density matrices.

    Args:
        rhos: dict mapping (mu, nu) -> (N, N) density matrix
        psi_0, psi_1: (N, 1) Fock kets

    Returns:
        Fe: real scalar
    """
    psi = [psi_0, psi_1]
    Fe = 0.0 + 0j
    for mu in range(2):
        for nu in range(2):
            Fe += (dag(psi[mu]) @ rhos[(mu, nu)] @ psi[nu]).squeeze()
    return float(jnp.real(Fe) / 4.0)


def simulate_rounds(recovery_ops, loss_ops, psi_0, psi_1, n_rounds):
    """
    Simulate n_rounds of (loss -> recovery) and compute Fe after each round.

    Each round:
      rho -> E(rho) -> R(E(rho))
    where E is the loss channel and R is the recovery channel.

    For entanglement fidelity, we track the 4 density matrix elements
    rho_{mu,nu} = (R o E)^n (|mu><nu|) independently.

    Args:
        recovery_ops: (K_R, N, N) recovery Kraus operators, or None for identity
        loss_ops: (K_E, N, N) loss Kraus operators
        psi_0, psi_1: (N, 1) normalized Fock kets
        n_rounds: number of (loss -> recovery) cycles

    Returns:
        fe_history: list of Fe values, length n_rounds
    """
    psi = [psi_0, psi_1]

    # Initialize: rho_{mu,nu} = |mu><nu|
    rhos = {}
    for mu in range(2):
        for nu in range(2):
            rhos[(mu, nu)] = psi[mu] @ dag(psi[nu])

    fe_history = []
    for _ in range(n_rounds):
        for mu in range(2):
            for nu in range(2):
                # Apply loss
                rho_after_loss = apply_kraus_map_nonorm(loss_ops, rhos[(mu, nu)])
                # Apply recovery
                if recovery_ops is not None:
                    rhos[(mu, nu)] = apply_kraus_map_nonorm(
                        recovery_ops, rho_after_loss)
                else:
                    rhos[(mu, nu)] = rho_after_loss

        fe_history.append(compute_fe_from_rhos(rhos, psi_0, psi_1))

    return fe_history


def simulate_rounds_jit(recovery_ops, loss_ops, psi_0, psi_1, n_rounds):
    """
    JIT-accelerated version of simulate_rounds using jax.lax.fori_loop.

    Stacks the 4 rho_{mu,nu} matrices into a (4, N, N) array for
    vectorized Kraus map application.
    """
    psi = [psi_0, psi_1]

    # Stack initial density matrices: rho[idx] where idx = 2*mu + nu
    N = psi_0.shape[0]
    rho_stack = jnp.zeros((4, N, N), dtype=jnp.complex64)
    for mu in range(2):
        for nu in range(2):
            idx = 2 * mu + nu
            rho_stack = rho_stack.at[idx].set(
                (psi[mu] @ dag(psi[nu])).squeeze()
                if psi[mu].ndim > 1
                else jnp.outer(psi[mu], jnp.conj(psi[nu]))
            )

    def apply_one_round(rho_stack, _):
        def apply_to_single(rho):
            after_loss = apply_kraus_map_nonorm(loss_ops, rho)
            if recovery_ops is not None:
                return apply_kraus_map_nonorm(recovery_ops, after_loss)
            return after_loss
        return jax.vmap(apply_to_single)(rho_stack), None

    # Collect Fe at each round using scan
    def scan_body(rho_stack, _):
        rho_next = jax.vmap(
            lambda rho: apply_kraus_map_nonorm(loss_ops, rho)
        )(rho_stack)
        if recovery_ops is not None:
            rho_next = jax.vmap(
                lambda rho: apply_kraus_map_nonorm(recovery_ops, rho)
            )(rho_next)

        # Compute Fe
        Fe = 0.0 + 0j
        for mu in range(2):
            for nu in range(2):
                idx = 2 * mu + nu
                Fe += (dag(psi[mu]) @ rho_next[idx] @ psi[nu]).squeeze()
        Fe = jnp.real(Fe) / 4.0

        return rho_next, Fe

    _, fe_array = jax.lax.scan(scan_body, rho_stack, None, length=n_rounds)
    return fe_array


# ============================================================
# MAIN DRIVER
# ============================================================

def run_multi_round(
    gammas=None,
    n_rounds=20,
    npz_path="results/cmaes_recovery_params.npz",
    Delta=0.3,
    N_trunc=3,
    loss_rank=10,
    save_dir="results",
):
    """
    Run multi-round stabilization simulation for all methods and gammas.
    """
    if gammas is None:
        gammas = [0.03, 0.05, 0.07, 0.10, 0.15]

    print("=" * 74)
    print("  Multi-Round Continuous Stabilization Simulation")
    print("=" * 74)
    print(f"  Rounds: {n_rounds}")
    print(f"  Per-round gammas: {gammas}")
    print(f"  Delta={Delta}, N_trunc={N_trunc}, loss_rank={loss_rank}")
    print()

    # Build GKP states
    print("Building GKP states...", end=" ")
    sys.stdout.flush()
    logical_0, logical_1 = build_gkp_states(
        Delta=Delta, N_trunc=N_trunc, lattice="square")
    psi_0 = coherent_ket_to_fock(logical_0)
    psi_1 = coherent_ket_to_fock(logical_1)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(jnp.conj(psi_0).T @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(jnp.conj(psi_1).T @ psi_1).squeeze())
    print("done.")

    # Build SBS recovery (gamma-independent)
    print("Building SBS Kraus operators...", end=" ")
    sys.stdout.flush()
    sbs_kraus = build_sbs_kraus(Delta=Delta)
    print("done.")

    # Load CMA-ES parameters
    print(f"Loading CMA-ES parameters from {npz_path}...", end=" ")
    sys.stdout.flush()
    npz_data = np.load(npz_path, allow_pickle=True)
    N_depth = int(npz_data["N_depth"])
    N_l = 2 ** N_depth
    print(f"N_depth={N_depth}, N_l={N_l}.")

    # Store all results
    all_results = {}
    methods = ["identity", "sbs", "cmaes", "transpose"]

    for gamma in gammas:
        print(f"\n{'=' * 70}")
        print(f"  gamma = {gamma}  (total loss after {n_rounds} rounds: "
              f"{1 - (1-gamma)**n_rounds:.4f})")
        print(f"{'=' * 70}")
        t0 = time.time()

        # Build loss operators for this gamma
        loss_ops = make_pureloss_fock(gamma, rank=loss_rank)

        # Build transpose recovery for this gamma
        transpose_ops = make_transpose_for_pureloss(
            loss_ops, logical_0, logical_1)

        # Build CMA-ES recovery for this gamma
        gamma_key = f"{gamma:.2f}".replace(".", "p")
        params_key = f"params_gamma_{gamma_key}"
        if params_key in npz_data:
            params = jnp.array(npz_data[params_key])
            alpha, beta = g(params, N_l)
            cmaes_kraus = channel_from_b(alpha, beta)
        else:
            print(f"  WARNING: No CMA-ES params for gamma={gamma}, skipping")
            cmaes_kraus = None

        # Run simulations for each method
        gamma_results = {}

        # 1. Identity (no recovery)
        print("  Simulating: identity...", end=" ")
        sys.stdout.flush()
        fe_id = simulate_rounds(None, loss_ops, psi_0, psi_1, n_rounds)
        gamma_results["identity"] = fe_id
        print(f"done. Final Fe={fe_id[-1]:.6f}")

        # 2. SBS
        print("  Simulating: SBS...", end=" ")
        sys.stdout.flush()
        fe_sbs = simulate_rounds(sbs_kraus, loss_ops, psi_0, psi_1, n_rounds)
        gamma_results["sbs"] = fe_sbs
        print(f"done. Final Fe={fe_sbs[-1]:.6f}")

        # 3. CMA-ES
        if cmaes_kraus is not None:
            print("  Simulating: CMA-ES...", end=" ")
            sys.stdout.flush()
            fe_cma = simulate_rounds(
                cmaes_kraus, loss_ops, psi_0, psi_1, n_rounds)
            gamma_results["cmaes"] = fe_cma
            print(f"done. Final Fe={fe_cma[-1]:.6f}")

        # 4. Transpose
        print("  Simulating: transpose...", end=" ")
        sys.stdout.flush()
        fe_tr = simulate_rounds(
            transpose_ops, loss_ops, psi_0, psi_1, n_rounds)
        gamma_results["transpose"] = fe_tr
        print(f"done. Final Fe={fe_tr[-1]:.6f}")

        all_results[gamma] = gamma_results
        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s")

        # Print per-gamma table
        print(f"\n  {'Round':>5s}", end="")
        for method in methods:
            if method in gamma_results:
                print(f" | {method:>10s}", end="")
        print()
        print(f"  {'-'*5}", end="")
        for method in methods:
            if method in gamma_results:
                print(f"-+-{'-'*10}", end="")
        print()

        # Print selected rounds
        display_rounds = [0, 1, 2, 3, 4, 5, 7, 10, 15, 19]
        display_rounds = [r for r in display_rounds if r < n_rounds]
        for r in display_rounds:
            print(f"  {r+1:5d}", end="")
            for method in methods:
                if method in gamma_results:
                    print(f" | {gamma_results[method][r]:10.6f}", end="")
            print()

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print(f"\n\n{'=' * 74}")
    print("  SUMMARY: Fe after N rounds of (loss -> recovery)")
    print(f"{'=' * 74}")

    for n_display in [1, 5, 10, n_rounds]:
        if n_display > n_rounds:
            continue
        idx = n_display - 1
        print(f"\n  After {n_display} round(s):")
        print(f"  {'gamma':>8s}", end="")
        for method in methods:
            print(f" | {method:>10s}", end="")
        print()
        print(f"  {'-'*8}", end="")
        for method in methods:
            print(f"-+-{'-'*10}", end="")
        print()
        for gamma in gammas:
            print(f"  {gamma:8.3f}", end="")
            for method in methods:
                if method in all_results[gamma]:
                    print(f" | {all_results[gamma][method][idx]:10.6f}",
                          end="")
                else:
                    print(f" | {'---':>10s}", end="")
            print()

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    os.makedirs(save_dir, exist_ok=True)
    save_dict = {
        "gammas": np.array(gammas),
        "n_rounds": n_rounds,
        "methods": np.array(methods),
    }
    for gamma in gammas:
        g_key = f"{gamma:.2f}".replace(".", "p")
        for method in methods:
            if method in all_results[gamma]:
                save_dict[f"fe_{method}_gamma_{g_key}"] = np.array(
                    all_results[gamma][method])

    npz_out = os.path.join(save_dir, "multi_round_results.npz")
    np.savez(npz_out, **save_dict)
    print(f"\n  Results saved to: {npz_out}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Multi-round stabilization simulation")
    parser.add_argument("--rounds", type=int, default=20,
                        help="Number of (loss -> recovery) rounds")
    parser.add_argument("--gammas", type=float, nargs="+",
                        default=None,
                        help="Per-round gamma values")
    parser.add_argument("--npz", default="results/cmaes_recovery_params.npz",
                        help="Path to CMA-ES parameters NPZ")
    args = parser.parse_args()

    run_multi_round(
        gammas=args.gammas,
        n_rounds=args.rounds,
        npz_path=args.npz,
    )


if __name__ == "__main__":
    main()
