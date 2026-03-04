"""
tree_multiround_optimizer_claude.py

Binary tree structure (depth >= 2) optimized for multi-round performance.

Key idea: more Kraus operators (4, 8, 16...) from deeper trees may better
approximate the transpose channel and provide more stable multi-round recovery.

Combines:
- build_leaf_displacements() from coherent_tree_optimizer_claude.py
- optimize_cmaes_tree() from coherent_tree_optimizer_claude.py
- Multi-round simulation from multi_round_claude.py

Usage:
    python -m coherax.tree_multiround_optimizer_claude
    python -m coherax.tree_multiround_optimizer_claude --depth 3 --rounds 20
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import sys
import os
import argparse

from coherax.characteristic_jax_utils import (
    CoherentKet,
    g, channel_from_b,
    make_pureloss_fock, make_transpose_for_pureloss,
    apply_kraus_map_nonorm, dag,
    GKP_N, dqcoherent, gkp_coherent_dm,
    coherent_overlap, aOmegab, e_n1iaOmegab,
)
from coherax.coherent_tree_optimizer_claude import (
    build_leaf_displacements,
    entanglement_fidelity_displacement,
)
from coherax.transpose_channel_claude import (
    build_gkp_states, coherent_ket_to_fock,
)


# ============================================================
# MULTI-ROUND SIMULATION (adapted from multi_round_claude.py)
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


def multi_round_fe(recovery_ops, loss_ops, psi_0, psi_1, n_rounds):
    """
    Compute entanglement fidelity after n_rounds.

    Returns:
        Fe_N: Fe after n_rounds
    """
    fe_history = simulate_rounds(recovery_ops, loss_ops, psi_0, psi_1, n_rounds)
    return fe_history[-1] if fe_history else 1.0


# ============================================================
# TREE MULTIROUND OPTIMIZATION
# ============================================================

def tree_recovery_to_fock(all_params, tree_depth, N_l, N=GKP_N):
    """
    Convert tree parameters to Fock-basis Kraus operators.

    Args:
        all_params: (n_nodes, N_depth, 4) circuit parameters
        tree_depth: depth of binary tree
        N_l: 2^N_depth displacements per node
        N: Fock space dimension

    Returns:
        recovery_ops: (n_leaves, N, N) Kraus operators in Fock basis
    """
    alpha_leaves, beta_leaves = build_leaf_displacements(
        all_params, tree_depth, N_l
    )
    return channel_from_b(alpha_leaves, beta_leaves)


def optimize_tree_multiround(
    logical_0, logical_1, gamma,
    tree_depth=2,
    N_depth=5,
    n_rounds=5,
    popsize=None,
    maxiter=2000,
    sigma0=3.0,
    seed=42,
    verbose=True,
):
    """
    Optimize tree recovery for multi-round performance using CMA-ES.

    The loss function optimizes the entanglement fidelity after n_rounds
    of (loss -> recovery) cycles.

    Args:
        logical_0, logical_1: CoherentKet GKP logical states
        gamma: per-round loss parameter
        tree_depth: depth of binary tree (2^depth Kraus ops)
        N_depth: CD+R layers per tree node (N_l = 2^N_depth)
        n_rounds: number of rounds to optimize for
        popsize: CMA-ES population size (auto-scaled if None)
        maxiter: max CMA-ES generations
        sigma0: initial step size
        seed: random seed
        verbose: print progress

    Returns:
        best_params: (n_nodes, N_depth, 4) optimized parameters
        best_Fe_N: best Fe after n_rounds
        info: dict with optimization details
    """
    import cma

    n_nodes = (1 << tree_depth) - 1
    n_leaves = 1 << tree_depth
    N_l = 2 ** N_depth
    n_params = n_nodes * N_depth * 4
    d_half = float(jnp.sqrt(jnp.pi / 2))

    # Auto-scale population size based on parameter count
    if popsize is None:
        popsize = max(80, 4 * int(np.sqrt(n_params)))

    # Build Fock-basis states and loss operators
    psi_0 = coherent_ket_to_fock(logical_0)
    psi_1 = coherent_ket_to_fock(logical_1)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(jnp.conj(psi_0).T @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(jnp.conj(psi_1).T @ psi_1).squeeze())
    loss_ops = make_pureloss_fock(gamma, rank=10)

    def unpack(x_real):
        """Convert real vector to (n_nodes, N_depth, 4) complex params."""
        p = jnp.zeros((n_nodes, N_depth, 4), dtype=jnp.complex64)
        idx = 0
        for node in range(n_nodes):
            for layer in range(N_depth):
                p = p.at[node, layer, 0].set(
                    x_real[idx] + 1j * x_real[idx+1])
                p = p.at[node, layer, 1].set(x_real[idx+2])
                p = p.at[node, layer, 2].set(x_real[idx+3])
                idx += 4
        return p

    def objective(x):
        """Multi-round Fe loss (negated for minimization)."""
        params = unpack(np.array(x))
        recovery_ops = tree_recovery_to_fock(params, tree_depth, N_l)
        Fe_N = multi_round_fe(recovery_ops, loss_ops, psi_0, psi_1, n_rounds)
        return -float(Fe_N)

    # GKP-informed initial point for each node
    x0 = np.zeros(n_params)
    for node in range(n_nodes):
        base = node * N_depth * 4
        angle = node * np.pi / max(n_nodes, 1)
        x0[base] = d_half * np.cos(angle)
        x0[base+1] = d_half * np.sin(angle)
        x0[base+3] = np.pi/2

    # Compute baseline (identity recovery)
    Fe_id_1 = multi_round_fe(None, loss_ops, psi_0, psi_1, 1)
    Fe_id_N = multi_round_fe(None, loss_ops, psi_0, psi_1, n_rounds)

    if verbose:
        print(f"CMA-ES tree multiround: depth={tree_depth}, n_nodes={n_nodes}, "
              f"N_depth={N_depth}, params={n_params}, pop={popsize}")
        print(f"  Optimizing for {n_rounds} rounds")
        print(f"  Fe_id (1 round): {Fe_id_1:.6f}")
        print(f"  Fe_id ({n_rounds} rounds): {Fe_id_N:.6f}")
        sys.stdout.flush()

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'maxiter': maxiter, 'popsize': popsize,
        'verbose': -1, 'seed': seed, 'tolfun': 1e-9,
    })

    gen = 0
    best_ever = 0.0
    above_baseline_gen = None
    t_start = time.time()

    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)
        best_now = -es.result.fbest
        best_ever = max(best_ever, best_now)

        if above_baseline_gen is None and best_ever > Fe_id_N + 0.001:
            above_baseline_gen = gen
            if verbose:
                print(f"  ** ABOVE BASELINE at gen {gen}! "
                      f"Fe_N={best_ever:.6f} **")
                sys.stdout.flush()

        if verbose and gen % 50 == 0:
            elapsed = time.time() - t_start
            print(f"  gen {gen}: Fe_N={best_now:.6f} "
                  f"(ever={best_ever:.6f}) [{elapsed:.0f}s]")
            sys.stdout.flush()
        gen += 1

    best_Fe_N = -es.result.fbest
    elapsed = time.time() - t_start
    best_params = unpack(es.result.xbest)

    if verbose:
        print(f"  Done ({elapsed:.0f}s, {gen} gens): "
              f"Fe_N={best_Fe_N:.6f}, improvement={best_Fe_N-Fe_id_N:.6f}")
        sys.stdout.flush()

    return best_params, best_Fe_N, {
        'Fe_id_1': Fe_id_1,
        'Fe_id_N': Fe_id_N,
        'generations': gen,
        'elapsed': elapsed,
        'above_baseline_gen': above_baseline_gen,
        'n_nodes': n_nodes,
        'tree_depth': tree_depth,
        'n_rounds': n_rounds,
    }


def optimize_tree_singleround_cmaes(
    logical_0, logical_1, gamma,
    tree_depth=2,
    N_depth=5,
    popsize=None,
    maxiter=2000,
    sigma0=3.0,
    seed=42,
    verbose=True,
):
    """
    CMA-ES optimization for tree recovery (single-round objective).

    This is a wrapper around optimize_cmaes_tree from coherent_tree_optimizer
    but using Fock-basis evaluation for consistency with multi-round.

    Args:
        Same as optimize_tree_multiround but no n_rounds.

    Returns:
        best_params: (n_nodes, N_depth, 4) optimized parameters
        best_Fe: best single-round Fe
        info: dict with optimization details
    """
    import cma

    n_nodes = (1 << tree_depth) - 1
    n_leaves = 1 << tree_depth
    N_l = 2 ** N_depth
    n_params = n_nodes * N_depth * 4
    d_half = float(jnp.sqrt(jnp.pi / 2))

    if popsize is None:
        popsize = max(80, 4 * int(np.sqrt(n_params)))

    # Build Fock-basis states and loss operators
    psi_0 = coherent_ket_to_fock(logical_0)
    psi_1 = coherent_ket_to_fock(logical_1)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(jnp.conj(psi_0).T @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(jnp.conj(psi_1).T @ psi_1).squeeze())
    loss_ops = make_pureloss_fock(gamma, rank=10)

    def unpack(x_real):
        p = jnp.zeros((n_nodes, N_depth, 4), dtype=jnp.complex64)
        idx = 0
        for node in range(n_nodes):
            for layer in range(N_depth):
                p = p.at[node, layer, 0].set(
                    x_real[idx] + 1j * x_real[idx+1])
                p = p.at[node, layer, 1].set(x_real[idx+2])
                p = p.at[node, layer, 2].set(x_real[idx+3])
                idx += 4
        return p

    def objective(x):
        """Single-round Fe loss."""
        params = unpack(np.array(x))
        recovery_ops = tree_recovery_to_fock(params, tree_depth, N_l)
        Fe = multi_round_fe(recovery_ops, loss_ops, psi_0, psi_1, 1)
        return -float(Fe)

    x0 = np.zeros(n_params)
    for node in range(n_nodes):
        base = node * N_depth * 4
        angle = node * np.pi / max(n_nodes, 1)
        x0[base] = d_half * np.cos(angle)
        x0[base+1] = d_half * np.sin(angle)
        x0[base+3] = np.pi/2

    Fe_id = multi_round_fe(None, loss_ops, psi_0, psi_1, 1)

    if verbose:
        print(f"CMA-ES tree single-round: depth={tree_depth}, n_nodes={n_nodes}, "
              f"N_depth={N_depth}, params={n_params}, pop={popsize}")
        print(f"  Fe_id={Fe_id:.6f}")
        sys.stdout.flush()

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'maxiter': maxiter, 'popsize': popsize,
        'verbose': -1, 'seed': seed, 'tolfun': 1e-9,
    })

    gen = 0
    best_ever = 0.0
    above_baseline_gen = None
    t_start = time.time()

    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)
        best_now = -es.result.fbest
        best_ever = max(best_ever, best_now)

        if above_baseline_gen is None and best_ever > Fe_id + 0.001:
            above_baseline_gen = gen
            if verbose:
                print(f"  ** ABOVE BASELINE at gen {gen}! "
                      f"Fe={best_ever:.6f} **")
                sys.stdout.flush()

        if verbose and gen % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  gen {gen}: Fe={best_now:.6f} "
                  f"(ever={best_ever:.6f}) [{elapsed:.0f}s]")
            sys.stdout.flush()
        gen += 1

    best_Fe = -es.result.fbest
    elapsed = time.time() - t_start
    best_params = unpack(es.result.xbest)

    if verbose:
        print(f"  Done ({elapsed:.0f}s, {gen} gens): "
              f"Fe={best_Fe:.6f}, improvement={best_Fe-Fe_id:.6f}")
        sys.stdout.flush()

    return best_params, best_Fe, {
        'Fe_id': Fe_id,
        'generations': gen,
        'elapsed': elapsed,
        'above_baseline_gen': above_baseline_gen,
        'n_nodes': n_nodes,
        'tree_depth': tree_depth,
    }


# ============================================================
# COMPARISON AND EVALUATION
# ============================================================

def evaluate_multiround_trajectory(
    params, tree_depth, N_depth, logical_0, logical_1, gamma,
    n_rounds=20, verbose=True,
):
    """
    Evaluate multi-round Fe trajectory for given tree parameters.

    Args:
        params: (n_nodes, N_depth, 4) tree parameters
        tree_depth, N_depth: tree structure
        logical_0, logical_1: CoherentKet GKP logical states
        gamma: per-round loss parameter
        n_rounds: number of rounds to evaluate
        verbose: print progress

    Returns:
        fe_history: list of Fe values for each round
    """
    N_l = 2 ** N_depth

    psi_0 = coherent_ket_to_fock(logical_0)
    psi_1 = coherent_ket_to_fock(logical_1)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(jnp.conj(psi_0).T @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(jnp.conj(psi_1).T @ psi_1).squeeze())
    loss_ops = make_pureloss_fock(gamma, rank=10)

    recovery_ops = tree_recovery_to_fock(params, tree_depth, N_l)
    fe_history = simulate_rounds(recovery_ops, loss_ops, psi_0, psi_1, n_rounds)

    if verbose:
        n_leaves = 1 << tree_depth
        print(f"  Tree (depth={tree_depth}, {n_leaves} Kraus) trajectory:")
        for i, fe in enumerate(fe_history):
            if i < 5 or i == n_rounds - 1 or (i + 1) % 5 == 0:
                print(f"    Round {i+1}: Fe={fe:.6f}")

    return fe_history


def compare_tree_depths(
    logical_0, logical_1, gamma,
    tree_depths=(1, 2, 3),
    N_depth=5,
    n_rounds_optimize=5,
    n_rounds_evaluate=20,
    popsize_base=80,
    maxiter=1500,
    verbose=True,
):
    """
    Compare different tree depths for multi-round recovery.

    For each tree depth:
    1. Optimize for n_rounds_optimize rounds
    2. Evaluate trajectory over n_rounds_evaluate rounds

    Args:
        logical_0, logical_1: CoherentKet GKP logical states
        gamma: per-round loss parameter
        tree_depths: tuple of tree depths to compare
        N_depth: CD+R layers per node
        n_rounds_optimize: rounds to optimize for
        n_rounds_evaluate: rounds to evaluate
        popsize_base: base CMA-ES population size
        maxiter: max CMA-ES generations
        verbose: print progress

    Returns:
        results: dict mapping depth -> {params, info, trajectory}
    """
    results = {}

    # Build Fock states for identity baseline
    psi_0 = coherent_ket_to_fock(logical_0)
    psi_1 = coherent_ket_to_fock(logical_1)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(jnp.conj(psi_0).T @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(jnp.conj(psi_1).T @ psi_1).squeeze())
    loss_ops = make_pureloss_fock(gamma, rank=10)

    # Identity baseline
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Identity (no recovery) baseline")
        print(f"{'='*60}")
    fe_id = simulate_rounds(None, loss_ops, psi_0, psi_1, n_rounds_evaluate)
    results['identity'] = {'trajectory': fe_id}
    if verbose:
        for i in [0, 4, 9, 14, 19]:
            if i < len(fe_id):
                print(f"    Round {i+1}: Fe={fe_id[i]:.6f}")

    # Transpose channel baseline
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Transpose channel baseline")
        print(f"{'='*60}")
    transpose_ops = make_transpose_for_pureloss(loss_ops, logical_0, logical_1)
    fe_tr = simulate_rounds(transpose_ops, loss_ops, psi_0, psi_1, n_rounds_evaluate)
    results['transpose'] = {'trajectory': fe_tr}
    if verbose:
        for i in [0, 4, 9, 14, 19]:
            if i < len(fe_tr):
                print(f"    Round {i+1}: Fe={fe_tr[i]:.6f}")

    # Optimize and evaluate each tree depth
    for depth in tree_depths:
        n_nodes = (1 << depth) - 1
        n_leaves = 1 << depth
        n_params = n_nodes * N_depth * 4
        popsize = max(popsize_base, 4 * int(np.sqrt(n_params)))

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Tree depth={depth} ({n_leaves} Kraus, {n_nodes} nodes, "
                  f"{n_params} params)")
            print(f"{'='*60}")

        # Multiround optimization
        params, Fe_N, info = optimize_tree_multiround(
            logical_0, logical_1, gamma,
            tree_depth=depth,
            N_depth=N_depth,
            n_rounds=n_rounds_optimize,
            popsize=popsize,
            maxiter=maxiter,
            verbose=verbose,
        )

        # Evaluate full trajectory
        if verbose:
            print(f"\n  Evaluating trajectory over {n_rounds_evaluate} rounds...")
        trajectory = evaluate_multiround_trajectory(
            params, depth, N_depth, logical_0, logical_1, gamma,
            n_rounds=n_rounds_evaluate, verbose=verbose,
        )

        results[depth] = {
            'params': params,
            'info': info,
            'trajectory': trajectory,
        }

    return results


def print_comparison_table(results, n_rounds_display=None):
    """
    Print a comparison table of results.

    Args:
        results: dict from compare_tree_depths
        n_rounds_display: rounds to show (default: [1, 5, 10, 15, 20])
    """
    if n_rounds_display is None:
        n_rounds_display = [1, 5, 10, 15, 20]

    methods = ['identity', 'transpose'] + [k for k in results.keys()
                                            if isinstance(k, int)]

    print(f"\n{'='*74}")
    print("  COMPARISON: Fe after N rounds")
    print(f"{'='*74}")

    # Header
    print(f"  {'Round':>5s}", end="")
    for method in methods:
        if method == 'identity':
            label = 'identity'
        elif method == 'transpose':
            label = 'transpose'
        else:
            n_leaves = 1 << method
            label = f'tree-{n_leaves}K'
        print(f" | {label:>10s}", end="")
    print()

    # Divider
    print(f"  {'-'*5}", end="")
    for _ in methods:
        print(f"-+-{'-'*10}", end="")
    print()

    # Data rows
    for r in n_rounds_display:
        idx = r - 1
        print(f"  {r:5d}", end="")
        for method in methods:
            traj = results[method]['trajectory']
            if idx < len(traj):
                print(f" | {traj[idx]:10.6f}", end="")
            else:
                print(f" | {'---':>10s}", end="")
        print()


# ============================================================
# MAIN DRIVER
# ============================================================

def run_full_comparison(
    gamma=0.05,
    Delta=0.3,
    N_trunc=3,
    tree_depths=(1, 2, 3),
    N_depth=5,
    n_rounds_optimize=5,
    n_rounds_evaluate=20,
    maxiter=1500,
    save_dir="results",
):
    """
    Run full tree depth comparison and save results.
    """
    print("=" * 74)
    print("  Tree Multiround Optimizer - Depth Comparison")
    print("=" * 74)
    print(f"  gamma={gamma}, Delta={Delta}")
    print(f"  Tree depths: {tree_depths}")
    print(f"  N_depth={N_depth} (N_l={2**N_depth})")
    print(f"  Optimize for {n_rounds_optimize} rounds, evaluate for {n_rounds_evaluate}")
    print()

    # Build GKP states
    print("Building GKP states...", end=" ")
    sys.stdout.flush()
    logical_0, logical_1 = build_gkp_states(
        Delta=Delta, N_trunc=N_trunc, lattice="square")
    print("done.")

    # Run comparison
    results = compare_tree_depths(
        logical_0, logical_1, gamma,
        tree_depths=tree_depths,
        N_depth=N_depth,
        n_rounds_optimize=n_rounds_optimize,
        n_rounds_evaluate=n_rounds_evaluate,
        maxiter=maxiter,
        verbose=True,
    )

    # Print comparison table
    print_comparison_table(results)

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    gamma_key = f"{gamma:.2f}".replace(".", "p")
    save_dict = {
        "gamma": gamma,
        "Delta": Delta,
        "tree_depths": np.array(tree_depths),
        "N_depth": N_depth,
        "n_rounds_optimize": n_rounds_optimize,
        "n_rounds_evaluate": n_rounds_evaluate,
    }

    for method, data in results.items():
        if method == 'identity':
            save_dict[f"traj_identity"] = np.array(data['trajectory'])
        elif method == 'transpose':
            save_dict[f"traj_transpose"] = np.array(data['trajectory'])
        else:
            depth = method
            save_dict[f"traj_depth_{depth}"] = np.array(data['trajectory'])
            save_dict[f"params_depth_{depth}"] = np.array(data['params'])

    npz_out = os.path.join(save_dir, f"tree_multiround_gamma_{gamma_key}.npz")
    np.savez(npz_out, **save_dict)
    print(f"\n  Results saved to: {npz_out}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Tree multiround optimizer comparison")
    parser.add_argument("--gamma", type=float, default=0.05,
                        help="Per-round loss parameter")
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3],
                        help="Tree depths to compare")
    parser.add_argument("--N_depth", type=int, default=5,
                        help="CD+R layers per node (N_l = 2^N_depth)")
    parser.add_argument("--rounds_opt", type=int, default=5,
                        help="Rounds to optimize for")
    parser.add_argument("--rounds_eval", type=int, default=20,
                        help="Rounds to evaluate")
    parser.add_argument("--maxiter", type=int, default=1500,
                        help="Max CMA-ES generations")
    parser.add_argument("--save_dir", default="results",
                        help="Directory to save results")
    args = parser.parse_args()

    run_full_comparison(
        gamma=args.gamma,
        tree_depths=tuple(args.depths),
        N_depth=args.N_depth,
        n_rounds_optimize=args.rounds_opt,
        n_rounds_evaluate=args.rounds_eval,
        maxiter=args.maxiter,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
