"""
transpose_approx_optimizer_claude.py

Optimizes CD+R circuits to approximate the transpose channel recovery.

The key insight: the transpose channel provides optimal single-round recovery
and demonstrates stability in multi-round settings. By training CD+R circuits
to approximate the transpose channel (rather than directly maximizing Fe),
we may obtain circuits that inherit this multi-round stability.

Three objective options are implemented:
  1. Choi state fidelity: F(J_cma, J_transpose)
  2. Average output fidelity over codespace
  3. Kraus operator matching (Procrustes-style)

Uses CMA-ES for derivative-free optimization, as the landscape is multimodal.

Usage:
    python -m coherax.transpose_approx_optimizer_claude
    python -m coherax.transpose_approx_optimizer_claude --gamma 0.05 --N_depth 6
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import time
import sys
import os
import argparse
from functools import partial

from coherax.characteristic_jax_utils import (
    CoherentKet,
    g, channel_from_b, super_g,
    make_pureloss_fock, make_transpose_for_pureloss,
    apply_kraus_map_nonorm, dag,
    gkp_coherent_dm, GKP_N, dqdag, dqcoherent, dqtrace,
)
from coherax.transpose_channel_claude import (
    build_gkp_states, coherent_ket_to_fock,
    entanglement_fidelity, entanglement_fidelity_no_recovery,
)
from coherax.coherent_tree_optimizer_claude import (
    entanglement_fidelity_displacement,
)


# ============================================================
# CHOI STATE CONSTRUCTION
# ============================================================

def build_choi_state(kraus_ops, N=GKP_N):
    """
    Build the Choi state (Choi-Jamiolkowski representation) of a channel.

    J(E) = (I otimes E)(|Omega><Omega|) where |Omega> = (1/sqrt(N)) sum_i |i,i>

    For a channel with Kraus operators {K_k}, this is:
        J = sum_k (I otimes K_k) |Omega><Omega| (I otimes K_k)^dag
          = (1/N) sum_k sum_{i,j} |i><j| otimes K_k |i><j| K_k^dag

    Returns the Choi state as a (N^2, N^2) matrix, or equivalently
    as a (N, N, N, N) tensor J[i,j,k,l] = <i,k|J|j,l>.

    For efficiency with large N, we return the reduced form directly.
    """
    n_kraus = kraus_ops.shape[0]

    # Choi state: J = sum_k (I otimes K_k) @ Omega @ (I otimes K_k)^dag
    # where Omega = |omega><omega|, |omega> = (1/sqrt(N)) sum_i |i,i>
    #
    # J[a,b,c,d] = (1/N) sum_k K_k[c,a] * conj(K_k[d,b])
    #            = (1/N) sum_k K_k[c,a] * K_k^*[d,b]

    # This gives: J = (1/N) sum_k K_k otimes K_k^*
    # Reshaped as (N,N,N,N): J[a,b,c,d] = (1/N) sum_k K_k[c,a] K_k^*[d,b]

    J = jnp.zeros((N, N, N, N), dtype=jnp.complex64)
    for k in range(n_kraus):
        # K otimes K^* : (N,N,N,N) where [a,b,c,d] = K[c,a] * K^*[d,b]
        J += jnp.einsum('ca,db->abcd', kraus_ops[k], jnp.conj(kraus_ops[k]))

    return J / N


def choi_fidelity(J1, J2, N=GKP_N):
    """
    Compute fidelity between two Choi states.

    F(J1, J2) = (Tr[sqrt(sqrt(J1) J2 sqrt(J1))])^2

    For numerical stability, we reshape to (N^2, N^2) matrices.
    """
    J1_mat = J1.reshape(N*N, N*N)
    J2_mat = J2.reshape(N*N, N*N)

    # Make Hermitian
    J1_mat = (J1_mat + dag(J1_mat)) / 2
    J2_mat = (J2_mat + dag(J2_mat)) / 2

    # Compute fidelity via eigendecomposition
    # F = (Tr[sqrt(sqrt(J1) J2 sqrt(J1))])^2

    # sqrt(J1) via eigendecomposition
    eigvals1, eigvecs1 = jnp.linalg.eigh(J1_mat)
    eigvals1 = jnp.maximum(eigvals1, 0)  # Ensure non-negative
    sqrt_J1 = eigvecs1 @ jnp.diag(jnp.sqrt(eigvals1)) @ dag(eigvecs1)

    # sqrt(J1) @ J2 @ sqrt(J1)
    M = sqrt_J1 @ J2_mat @ sqrt_J1
    M = (M + dag(M)) / 2  # Ensure Hermitian

    # Trace of sqrt(M)
    eigvals_M, _ = jnp.linalg.eigh(M)
    eigvals_M = jnp.maximum(eigvals_M, 0)
    tr_sqrt = jnp.sum(jnp.sqrt(eigvals_M))

    return jnp.real(tr_sqrt ** 2)


def trace_distance_choi(J1, J2, N=GKP_N):
    """
    Compute trace distance between two Choi states.
    D(J1, J2) = 0.5 * Tr[|J1 - J2|]
    """
    J1_mat = J1.reshape(N*N, N*N)
    J2_mat = J2.reshape(N*N, N*N)

    diff = J1_mat - J2_mat
    diff = (diff + dag(diff)) / 2

    eigvals, _ = jnp.linalg.eigh(diff)
    return 0.5 * jnp.sum(jnp.abs(eigvals))


# ============================================================
# PROCRUSTES KRAUS MATCHING
# ============================================================

def procrustes_kraus_distance(K_cma, K_transpose):
    """
    Procrustes-corrected distance between two sets of Kraus operators.

    Finds the optimal unitary U that minimizes:
        sum_k ||K_cma[k] - U @ K_transpose[k]||^2

    This accounts for the freedom to apply a global unitary to Kraus ops.

    Uses the closed-form solution via SVD:
        M = sum_k K_transpose[k] @ K_cma[k]^dag
        U_opt = V @ W^dag  where M = V @ S @ W^dag
        dist_opt = sum_k (||K_cma[k]||^2 + ||K_transpose[k]||^2) - 2*Tr[S]
    """
    n_kraus = K_cma.shape[0]

    # Fixed terms
    fixed = (jnp.sum(jnp.abs(K_cma)**2) + jnp.sum(jnp.abs(K_transpose)**2))

    # Build M = sum_k K_transpose[k] @ K_cma[k]^dag
    M = jnp.zeros_like(K_cma[0])
    for k in range(n_kraus):
        M += K_transpose[k] @ dag(K_cma[k])

    # SVD and optimal distance
    s = jnp.linalg.svdvals(M)

    return jnp.real(fixed - 2.0 * jnp.sum(s))


# ============================================================
# AVERAGE OUTPUT FIDELITY
# ============================================================

def average_output_fidelity(K_cma, K_transpose, basis_states):
    """
    Average fidelity between outputs of two channels over a set of states.

    F_avg = (1/|S|) sum_{rho in S} F(R_cma(rho), R_transpose(rho))

    where F(rho, sigma) = Tr[sqrt(sqrt(rho) sigma sqrt(rho))]^2
    """
    n_states = len(basis_states)

    def output_fidelity(rho):
        out_cma = apply_kraus_map_nonorm(K_cma, rho)
        out_tr = apply_kraus_map_nonorm(K_transpose, rho)

        # Normalize
        out_cma = out_cma / jnp.maximum(dqtrace(out_cma), 1e-10)
        out_tr = out_tr / jnp.maximum(dqtrace(out_tr), 1e-10)

        # Fidelity via eigendecomposition
        out_cma = (out_cma + dag(out_cma)) / 2
        out_tr = (out_tr + dag(out_tr)) / 2

        eigvals1, eigvecs1 = jnp.linalg.eigh(out_cma)
        eigvals1 = jnp.maximum(eigvals1, 0)
        sqrt_cma = eigvecs1 @ jnp.diag(jnp.sqrt(eigvals1)) @ dag(eigvecs1)

        M = sqrt_cma @ out_tr @ sqrt_cma
        M = (M + dag(M)) / 2

        eigvals_M, _ = jnp.linalg.eigh(M)
        eigvals_M = jnp.maximum(eigvals_M, 0)
        tr_sqrt = jnp.sum(jnp.sqrt(eigvals_M))

        return jnp.real(tr_sqrt ** 2)

    fids = jnp.array([output_fidelity(rho) for rho in basis_states])
    return jnp.mean(fids)


# ============================================================
# CMA-ES OPTIMIZER
# ============================================================

def optimize_transpose_approx_cmaes(
    logical_0, logical_1, gamma,
    objective="choi",  # "choi", "procrustes", or "output"
    N_depth=6,
    T_depth=1,
    popsize=80,
    maxiter=2000,
    sigma0=3.0,
    n_restarts=5,
    loss_rank=10,
    verbose=True,
):
    """
    CMA-ES optimization to approximate the transpose channel.

    Args:
        logical_0, logical_1: CoherentKet GKP logical states
        gamma: loss parameter
        objective: "choi" for Choi fidelity, "procrustes" for Kraus matching,
                   "output" for average output fidelity
        N_depth: CD+R circuit depth per traceout round
        T_depth: number of traceout rounds
        popsize: CMA-ES population size
        maxiter: max generations per restart
        sigma0: initial step size
        n_restarts: number of random restarts
        loss_rank: number of loss Kraus operators
        verbose: print progress

    Returns:
        best_params: (T_depth, N_depth, 4) optimized parameters
        best_loss: final loss value
        info: dict with optimization details
    """
    import cma

    N_l = 2 ** N_depth
    N = GKP_N
    n_params = T_depth * N_depth * 4
    d_half = float(jnp.sqrt(jnp.pi / 2))

    # Build transpose channel (target)
    loss_ops = make_pureloss_fock(gamma, rank=loss_rank, N=N)
    transpose_ops = make_transpose_for_pureloss(loss_ops, logical_0, logical_1)

    # Build Fock-basis states for entanglement fidelity
    psi_0 = coherent_ket_to_fock(logical_0, N)
    psi_1 = coherent_ket_to_fock(logical_1, N)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(dag(psi_0) @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(dag(psi_1) @ psi_1).squeeze())

    # Build Choi state for transpose channel (if needed)
    if objective == "choi":
        J_transpose = build_choi_state(transpose_ops, N)

    # Build basis states for output fidelity (if needed)
    if objective == "output":
        # Use |0_L>, |1_L>, |+_L>, |+i_L> as test states
        rho_0 = psi_0 @ dag(psi_0)
        rho_1 = psi_1 @ dag(psi_1)
        plus_state = (psi_0 + psi_1) / jnp.sqrt(2)
        plus_state = plus_state / jnp.sqrt(jnp.real(dag(plus_state) @ plus_state).squeeze())
        rho_plus = plus_state @ dag(plus_state)
        plus_i_state = (psi_0 + 1j * psi_1) / jnp.sqrt(2)
        plus_i_state = plus_i_state / jnp.sqrt(jnp.real(dag(plus_i_state) @ plus_i_state).squeeze())
        rho_plus_i = plus_i_state @ dag(plus_i_state)
        basis_states = [rho_0, rho_1, rho_plus, rho_plus_i]

    def unpack(x_real):
        """Convert real parameter vector to complex (T_depth, N_depth, 4) params."""
        p = jnp.zeros((T_depth, N_depth, 4), dtype=jnp.complex64)
        idx = 0
        for t in range(T_depth):
            for layer in range(N_depth):
                p = p.at[t, layer, 0].set(x_real[idx] + 1j * x_real[idx+1])
                p = p.at[t, layer, 1].set(x_real[idx+2])
                p = p.at[t, layer, 2].set(x_real[idx+3])
                idx += 4
        return p

    def build_recovery_ops(p_complex):
        """Build Fock-basis Kraus operators from circuit parameters."""
        if T_depth == 1:
            alpha, beta = g(p_complex[0], N_l)
        else:
            alpha, beta = super_g(p_complex, N_l=N_l, T=T_depth)
        return channel_from_b(alpha, beta)

    def objective_fn(x_real):
        """Objective to minimize (negative of metric we want to maximize)."""
        p = unpack(np.array(x_real))
        recovery_ops = build_recovery_ops(p)

        if objective == "choi":
            J_cma = build_choi_state(recovery_ops, N)
            fid = choi_fidelity(J_cma, J_transpose, N)
            return float(1.0 - fid)

        elif objective == "procrustes":
            dist = procrustes_kraus_distance(recovery_ops, transpose_ops)
            return float(dist)

        elif objective == "output":
            fid = average_output_fidelity(recovery_ops, transpose_ops, basis_states)
            return float(1.0 - fid)

        else:
            raise ValueError(f"Unknown objective: {objective}")

    # Identity baseline
    id_params = np.zeros(n_params)
    id_loss = objective_fn(id_params)

    # Compute transpose channel Fe for reference
    Fe_transpose = float(entanglement_fidelity(
        transpose_ops, loss_ops, psi_0, psi_1))

    if verbose:
        print(f"\n  Transpose approximation via {objective} objective")
        print(f"  gamma={gamma}, N_depth={N_depth}, T_depth={T_depth}")
        print(f"  n_params={n_params}, popsize={popsize}")
        print(f"  Identity loss: {id_loss:.6f}")
        print(f"  Fe_transpose (target): {Fe_transpose:.6f}")
        sys.stdout.flush()

    best_loss = id_loss
    best_x = np.zeros(n_params)
    trials = []
    t_total = time.time()

    for trial in range(n_restarts):
        # GKP-informed initial point
        x0 = np.zeros(n_params)
        for t in range(T_depth):
            base = t * N_depth * 4
            x0[base] = d_half
            x0[base + 3] = np.pi / 2
            if N_depth > 1:
                x0[base + 5] = d_half
                x0[base + 7] = np.pi / 2

        # Add some randomness for diversity
        x0 += np.random.randn(n_params) * 0.5

        es = cma.CMAEvolutionStrategy(x0, sigma0, {
            'maxiter': maxiter,
            'popsize': popsize,
            'verbose': -1,
            'seed': trial,
            'tolfun': 1e-9,
        })

        gen = 0
        t0 = time.time()
        while not es.stop():
            solutions = es.ask()
            fitnesses = [objective_fn(x) for x in solutions]
            es.tell(solutions, fitnesses)
            gen += 1

        trial_loss = es.result.fbest
        elapsed = time.time() - t0
        improved = trial_loss < id_loss - 0.001

        trials.append({
            'seed': trial,
            'loss': trial_loss,
            'gens': gen,
            'time': elapsed,
            'improved': improved,
        })

        if verbose:
            flag = ' ***' if improved else ''
            print(f"    trial {trial:2d}: loss={trial_loss:.6f} "
                  f"({gen} gens, {elapsed:.0f}s){flag}")
            sys.stdout.flush()

        if trial_loss < best_loss:
            best_loss = trial_loss
            best_x = es.result.xbest.copy()

    elapsed_total = time.time() - t_total
    best_params = unpack(best_x)
    n_improved = sum(1 for t in trials if t['improved'])

    if verbose:
        print(f"\n  Best loss: {best_loss:.6f}")
        print(f"  Improved: {n_improved}/{n_restarts} trials")
        print(f"  Total time: {elapsed_total:.0f}s")
        sys.stdout.flush()

    return best_params, best_loss, {
        'objective': objective,
        'id_loss': id_loss,
        'Fe_transpose': Fe_transpose,
        'trials': trials,
        'n_improved': n_improved,
        'total_time': elapsed_total,
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_recovery(
    params,
    logical_0, logical_1, gamma,
    N_depth=6,
    T_depth=1,
    n_rounds=20,
    loss_rank=10,
    verbose=True,
):
    """
    Evaluate the optimized recovery circuit.

    Returns:
        dict with single-round Fe, multi-round Fe trajectory, channel distances
    """
    N_l = 2 ** N_depth
    N = GKP_N

    # Build operators
    loss_ops = make_pureloss_fock(gamma, rank=loss_rank, N=N)
    transpose_ops = make_transpose_for_pureloss(loss_ops, logical_0, logical_1)

    if T_depth == 1:
        alpha, beta = g(params[0] if params.ndim == 3 else params, N_l)
    else:
        alpha, beta = super_g(params, N_l=N_l, T=T_depth)
    cma_ops = channel_from_b(alpha, beta)

    # Build Fock states
    psi_0 = coherent_ket_to_fock(logical_0, N)
    psi_1 = coherent_ket_to_fock(logical_1, N)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(dag(psi_0) @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(dag(psi_1) @ psi_1).squeeze())

    # Single-round Fe
    Fe_none = float(entanglement_fidelity_no_recovery(loss_ops, psi_0, psi_1))
    Fe_cma = float(entanglement_fidelity(cma_ops, loss_ops, psi_0, psi_1))
    Fe_transpose = float(entanglement_fidelity(transpose_ops, loss_ops, psi_0, psi_1))

    # Channel distances
    J_cma = build_choi_state(cma_ops, N)
    J_transpose = build_choi_state(transpose_ops, N)
    choi_fid = float(choi_fidelity(J_cma, J_transpose, N))
    trace_dist = float(trace_distance_choi(J_cma, J_transpose, N))
    procrustes_dist = float(procrustes_kraus_distance(cma_ops, transpose_ops))

    # Multi-round simulation
    def simulate_rounds(recovery_ops, n_rounds):
        psi = [psi_0, psi_1]
        rhos = {}
        for mu in range(2):
            for nu in range(2):
                rhos[(mu, nu)] = psi[mu] @ dag(psi[nu])

        fe_history = []
        for _ in range(n_rounds):
            for mu in range(2):
                for nu in range(2):
                    after_loss = apply_kraus_map_nonorm(loss_ops, rhos[(mu, nu)])
                    if recovery_ops is not None:
                        rhos[(mu, nu)] = apply_kraus_map_nonorm(
                            recovery_ops, after_loss)
                    else:
                        rhos[(mu, nu)] = after_loss

            Fe = 0.0 + 0j
            for mu in range(2):
                for nu in range(2):
                    Fe += (dag(psi[mu]) @ rhos[(mu, nu)] @ psi[nu]).squeeze()
            fe_history.append(float(jnp.real(Fe) / 4.0))

        return fe_history

    fe_none_multi = simulate_rounds(None, n_rounds)
    fe_cma_multi = simulate_rounds(cma_ops, n_rounds)
    fe_transpose_multi = simulate_rounds(transpose_ops, n_rounds)

    if verbose:
        print(f"\n  Evaluation for gamma={gamma}:")
        print(f"    Single-round Fe:")
        print(f"      Identity:   {Fe_none:.6f}")
        print(f"      CMA-approx: {Fe_cma:.6f}")
        print(f"      Transpose:  {Fe_transpose:.6f}")
        print(f"      Gap to transpose: {Fe_transpose - Fe_cma:.6f}")
        print(f"    Channel distance to transpose:")
        print(f"      Choi fidelity:    {choi_fid:.6f}")
        print(f"      Trace distance:   {trace_dist:.6f}")
        print(f"      Procrustes dist:  {procrustes_dist:.6f}")
        print(f"    Multi-round Fe (after {n_rounds} rounds):")
        print(f"      Identity:   {fe_none_multi[-1]:.6f}")
        print(f"      CMA-approx: {fe_cma_multi[-1]:.6f}")
        print(f"      Transpose:  {fe_transpose_multi[-1]:.6f}")
        sys.stdout.flush()

    return {
        'gamma': gamma,
        'Fe_none': Fe_none,
        'Fe_cma': Fe_cma,
        'Fe_transpose': Fe_transpose,
        'choi_fidelity': choi_fid,
        'trace_distance': trace_dist,
        'procrustes_distance': procrustes_dist,
        'fe_none_multi': fe_none_multi,
        'fe_cma_multi': fe_cma_multi,
        'fe_transpose_multi': fe_transpose_multi,
    }


# ============================================================
# COMPARISON OF OBJECTIVES
# ============================================================

def compare_objectives(
    gamma=0.05,
    Delta=0.3,
    N_trunc=3,
    N_depth=6,
    T_depth=1,
    popsize=80,
    maxiter=1000,
    n_restarts=3,
    n_rounds=20,
    verbose=True,
):
    """
    Compare different objectives for transpose approximation.

    Runs CMA-ES with each objective and evaluates the resulting circuits.
    """
    print("=" * 70)
    print("  Transpose Approximation: Objective Comparison")
    print("=" * 70)
    print(f"  gamma={gamma}, Delta={Delta}, N_depth={N_depth}, T_depth={T_depth}")
    print()

    # Build GKP states
    logical_0, logical_1 = build_gkp_states(
        Delta=Delta, N_trunc=N_trunc, lattice="square")

    objectives = ["choi", "procrustes", "output"]
    results = {}

    for obj in objectives:
        print(f"\n{'=' * 60}")
        print(f"  Objective: {obj}")
        print(f"{'=' * 60}")

        params, loss, opt_info = optimize_transpose_approx_cmaes(
            logical_0, logical_1, gamma,
            objective=obj,
            N_depth=N_depth,
            T_depth=T_depth,
            popsize=popsize,
            maxiter=maxiter,
            n_restarts=n_restarts,
            verbose=verbose,
        )

        eval_result = evaluate_recovery(
            params, logical_0, logical_1, gamma,
            N_depth=N_depth, T_depth=T_depth,
            n_rounds=n_rounds, verbose=verbose,
        )

        results[obj] = {
            'params': params,
            'opt_loss': loss,
            'opt_info': opt_info,
            'eval': eval_result,
        }

    # Summary table
    print(f"\n\n{'=' * 70}")
    print("  SUMMARY: Comparison of Objectives")
    print(f"{'=' * 70}")
    print(f"\n  {'Objective':>12s} | {'Fe_cma':>10s} | {'ChF':>8s} | "
          f"{'TrD':>8s} | {'Fe_multi':>10s}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")

    for obj in objectives:
        r = results[obj]['eval']
        print(f"  {obj:>12s} | {r['Fe_cma']:10.6f} | "
              f"{r['choi_fidelity']:8.4f} | {r['trace_distance']:8.4f} | "
              f"{r['fe_cma_multi'][-1]:10.6f}")

    # Print transpose reference
    print(f"  {'transpose':>12s} | {results['choi']['eval']['Fe_transpose']:10.6f} | "
          f"{'1.0000':>8s} | {'0.0000':>8s} | "
          f"{results['choi']['eval']['fe_transpose_multi'][-1]:10.6f}")

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optimize CD+R to approximate transpose channel")
    parser.add_argument("--gamma", type=float, default=0.05,
                        help="Loss parameter")
    parser.add_argument("--Delta", type=float, default=0.3,
                        help="GKP envelope parameter")
    parser.add_argument("--N_depth", type=int, default=6,
                        help="CD+R circuit depth")
    parser.add_argument("--T_depth", type=int, default=1,
                        help="Number of traceout rounds")
    parser.add_argument("--objective", type=str, default="choi",
                        choices=["choi", "procrustes", "output", "compare"],
                        help="Objective function")
    parser.add_argument("--popsize", type=int, default=80,
                        help="CMA-ES population size")
    parser.add_argument("--maxiter", type=int, default=1000,
                        help="CMA-ES max iterations")
    parser.add_argument("--restarts", type=int, default=3,
                        help="Number of CMA-ES restarts")
    parser.add_argument("--rounds", type=int, default=20,
                        help="Multi-round simulation length")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Directory to save results")
    args = parser.parse_args()

    if args.objective == "compare":
        results = compare_objectives(
            gamma=args.gamma,
            Delta=args.Delta,
            N_depth=args.N_depth,
            T_depth=args.T_depth,
            popsize=args.popsize,
            maxiter=args.maxiter,
            n_restarts=args.restarts,
            n_rounds=args.rounds,
        )

        # Save results
        os.makedirs(args.save_dir, exist_ok=True)
        save_dict = {
            'gamma': args.gamma,
            'Delta': args.Delta,
            'N_depth': args.N_depth,
            'T_depth': args.T_depth,
        }
        for obj in ['choi', 'procrustes', 'output']:
            r = results[obj]
            save_dict[f'params_{obj}'] = np.array(r['params'])
            save_dict[f'Fe_cma_{obj}'] = r['eval']['Fe_cma']
            save_dict[f'choi_fid_{obj}'] = r['eval']['choi_fidelity']
            save_dict[f'fe_multi_{obj}'] = np.array(r['eval']['fe_cma_multi'])

        save_dict['Fe_transpose'] = results['choi']['eval']['Fe_transpose']
        save_dict['fe_transpose_multi'] = np.array(
            results['choi']['eval']['fe_transpose_multi'])

        npz_path = os.path.join(
            args.save_dir,
            f"transpose_approx_gamma_{args.gamma:.2f}.npz".replace(".", "p", 1))
        np.savez(npz_path, **save_dict)
        print(f"\n  Results saved to: {npz_path}")

    else:
        # Single objective optimization
        print("=" * 70)
        print("  Transpose Channel Approximation Optimizer")
        print("=" * 70)
        print(f"  gamma={args.gamma}, Delta={args.Delta}")
        print(f"  N_depth={args.N_depth}, T_depth={args.T_depth}")
        print(f"  Objective: {args.objective}")
        print()

        logical_0, logical_1 = build_gkp_states(
            Delta=args.Delta, N_trunc=3, lattice="square")

        params, loss, opt_info = optimize_transpose_approx_cmaes(
            logical_0, logical_1, args.gamma,
            objective=args.objective,
            N_depth=args.N_depth,
            T_depth=args.T_depth,
            popsize=args.popsize,
            maxiter=args.maxiter,
            n_restarts=args.restarts,
            verbose=True,
        )

        eval_result = evaluate_recovery(
            params, logical_0, logical_1, args.gamma,
            N_depth=args.N_depth, T_depth=args.T_depth,
            n_rounds=args.rounds, verbose=True,
        )

        # Save results
        os.makedirs(args.save_dir, exist_ok=True)
        save_dict = {
            'gamma': args.gamma,
            'Delta': args.Delta,
            'N_depth': args.N_depth,
            'T_depth': args.T_depth,
            'objective': args.objective,
            'params': np.array(params),
            'opt_loss': loss,
            'Fe_cma': eval_result['Fe_cma'],
            'Fe_transpose': eval_result['Fe_transpose'],
            'choi_fidelity': eval_result['choi_fidelity'],
            'fe_cma_multi': np.array(eval_result['fe_cma_multi']),
            'fe_transpose_multi': np.array(eval_result['fe_transpose_multi']),
        }

        npz_path = os.path.join(
            args.save_dir,
            f"transpose_approx_{args.objective}_gamma_{args.gamma:.2f}.npz".replace(
                ".", "p", 1))
        np.savez(npz_path, **save_dict)
        print(f"\n  Results saved to: {npz_path}")


if __name__ == "__main__":
    main()
