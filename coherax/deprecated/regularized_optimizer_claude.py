"""
regularized_optimizer_claude.py

Regularized CD+R circuit optimization with identity-bias to prevent over-correction.

The key insight: at high gamma (strong loss), the optimal single-round recovery
may introduce phase-space distortions that compound destructively over multiple
rounds. Regularizing toward identity prevents this by penalizing circuit
complexity.

Three regularization strategies:
  1. Parameter norm: ||params||^2 pushes displacements and rotations to zero
  2. Kraus deviation: ||K - I|| where K is the synthesized Kraus operator
  3. Channel overlap: Tr(R I) / sqrt(Tr(R R) Tr(I)) measures channel similarity

CMA-ES is used for optimization since the landscape is highly multimodal.

Pipeline:
  1. Set up regularized loss function
  2. Sweep lambda values to find optimal regularization strength
  3. Compare single-round vs multi-round fidelity
  4. Identify gamma regimes where regularization helps
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from functools import partial
from jaxtyping import Array
import time
import sys

from coherax.characteristic_jax_utils import (
    CoherentKet,
    coherent_overlap,
    aOmegab,
    e_n1iaOmegab,
    dag,
    g,
    GKP_N,
    dqcoherent,
    make_pureloss_fock,
    apply_kraus_map_nonorm,
    channel_from_b,
    gkp_coherent_dm,
)


# ============================================================
# ENTANGLEMENT FIDELITY (from coherent_tree_optimizer_claude.py)
# ============================================================

def entanglement_fidelity_displacement(alpha, beta, c_0, d_0, c_1, d_1, gamma):
    """
    Compute entanglement fidelity directly in the coherent basis.

    Fe = (1/4) sum_{mu,nu} sum_k <psi_mu| R_k E(|psi_mu><psi_nu|) R_k^dag |psi_nu>

    where R_k = sum_j alpha[k,j] D(beta[k,j]) are recovery operators and
    E is the pure loss channel with parameter gamma.
    """
    t = jnp.sqrt(1.0 - gamma)
    r = jnp.sqrt(gamma)
    n_kraus = alpha.shape[0]

    cs = [c_0, c_1]
    ds = [d_0, d_1]

    # Precompute env overlaps for all (mu,nu) pairs
    env = {}
    for mu in range(2):
        env[mu] = {}
        for nu in range(2):
            env[mu][nu] = coherent_overlap(
                r * ds[nu].reshape(-1, 1),
                r * ds[mu].reshape(1, -1),
            )

    Fe = 0.0 + 0j

    for k in range(n_kraus):
        L = {}
        for mu in range(2):
            A_mu = ds[mu].shape[0]
            td_mu = t * ds[mu]

            phase = jnp.exp(-1j * aOmegab(
                beta[k, :, None],
                td_mu[None, :],
            ))

            shifted = beta[k, :, None] + td_mu[None, :]

            ovlp = coherent_overlap(
                ds[mu][:, None, None],
                shifted[None, :, :],
            )

            L[mu] = jnp.einsum(
                'p,j,ja,pja->a',
                jnp.conj(cs[mu]), alpha[k], phase, ovlp,
            )

        for mu in range(2):
            for nu in range(2):
                v_mu = cs[mu] * L[mu]
                v_nu = cs[nu] * L[nu]
                Fe += jnp.conj(v_nu) @ env[mu][nu] @ v_mu

    return jnp.real(Fe) / 4.0


# ============================================================
# REGULARIZATION TERMS
# ============================================================

def param_norm_regularization(params):
    """
    Regularize by penalizing parameter magnitude.

    Identity channel corresponds to params = 0 (no displacements, no rotations).

    Args:
        params: (N_depth, 4) circuit parameters [d, phi, theta, gamma_rot]

    Returns:
        reg: scalar regularization penalty
    """
    # Displacement magnitude (complex)
    d_norm = jnp.sum(jnp.abs(params[:, 0])**2)

    # Angle magnitude (phi and theta) - take real part since params can be complex
    angle_norm = jnp.sum(jnp.real(params[:, 1])**2 + jnp.real(params[:, 2])**2)

    return d_norm + angle_norm


def kraus_identity_deviation(alpha, beta):
    """
    Compute deviation of Kraus operators from identity.

    For identity channel: K_0 = I, K_k = 0 for k > 0
    In displacement rep: alpha = [[1]], beta = [[0]]

    We measure: sum_k ||K_k - delta_{k0} I||^2

    In coherent basis, this becomes an overlap integral.

    Args:
        alpha: (n_kraus, N_disp) Kraus coefficients
        beta: (n_kraus, N_disp) displacement positions

    Returns:
        deviation: scalar measuring distance from identity
    """
    n_kraus = alpha.shape[0]

    deviation = 0.0

    for k in range(n_kraus):
        # ||K_k||^2 = sum_{j,l} alpha_j* alpha_l <D(beta_j)|D(beta_l)>
        #           = sum_{j,l} alpha_j* alpha_l exp(-|beta_j - beta_l|^2/2)
        alpha_k = alpha[k]
        beta_k = beta[k]

        # Compute Tr(K_k^dag K_k)
        alpha_outer = jnp.conj(alpha_k[:, None]) * alpha_k[None, :]
        beta_diff = beta_k[:, None] - beta_k[None, :]
        overlap = jnp.exp(-0.5 * jnp.abs(beta_diff)**2)
        trace_kk = jnp.real(jnp.sum(alpha_outer * overlap))

        if k == 0:
            # For k=0, we want K_0 close to I
            # ||K_0 - I||^2 = ||K_0||^2 - 2 Re(Tr(K_0)) + N
            # Tr(K_0) = sum_j alpha_j (N is dimension, but in coherent basis,
            # we approximate Tr(D(0)) = 1)
            trace_k0 = jnp.sum(alpha_k * jnp.exp(-0.5 * jnp.abs(beta_k)**2))
            deviation += trace_kk - 2.0 * jnp.real(trace_k0) + 1.0
        else:
            # For k > 0, we want K_k = 0
            deviation += trace_kk

    return deviation


def channel_identity_overlap(alpha, beta, n_test_points=20):
    """
    Measure overlap between recovery channel and identity via characteristic function.

    Identity channel: C_I(u) = 1 for all u
    Recovery channel: C_R(u) = sum_k sum_{j,l} alpha_kj* alpha_kl
                                * exp(-|u + beta_kl - beta_kj|^2/2) * phases

    Overlap = integral |C_R(u) - C_I(u)|^2 du

    We approximate with a discrete sum over test points.

    Args:
        alpha: (n_kraus, N_disp) Kraus coefficients
        beta: (n_kraus, N_disp) displacement positions
        n_test_points: number of test points per axis

    Returns:
        deviation: scalar measuring distance from identity
    """
    # Test points in phase space
    x = jnp.linspace(-3, 3, n_test_points)
    y = jnp.linspace(-3, 3, n_test_points)
    X, Y = jnp.meshgrid(x, y)
    u_test = (X + 1j * Y).reshape(-1)  # (n_test^2,)

    n_kraus = alpha.shape[0]
    N_disp = alpha.shape[1]

    # Compute characteristic function at test points
    # C_R(u) = sum_k sum_{j,l} conj(alpha_kj) * alpha_kl
    #          * exp(-|u + beta_kl - beta_kj|^2/2) * exp(i * phase)

    deviation = 0.0

    for i, u in enumerate(u_test):
        C_R = 0.0 + 0j
        for k in range(n_kraus):
            alpha_k = alpha[k]  # (N_disp,)
            beta_k = beta[k]    # (N_disp,)

            # beta_diff[j,l] = beta_l - beta_j
            beta_diff = beta_k[None, :] - beta_k[:, None]

            # envelope[j,l] = exp(-|u + beta_l - beta_j|^2/2)
            envelope = jnp.exp(-0.5 * jnp.abs(u + beta_diff)**2)

            # phase[j,l] = exp(i * (aOmegab(u, beta_l + beta_j) + aOmegab(beta_l, beta_j)))
            phase = jnp.exp(1j * (aOmegab(u, beta_k[None, :] + beta_k[:, None])
                                   + aOmegab(beta_k[None, :], beta_k[:, None])))

            # Sum over j,l
            alpha_jl = jnp.conj(alpha_k[:, None]) * alpha_k[None, :]
            C_R += jnp.sum(alpha_jl * envelope * phase)

        # |C_R(u) - 1|^2
        deviation += jnp.abs(C_R - 1.0)**2

    # Normalize by number of test points
    return jnp.real(deviation) / len(u_test)


# ============================================================
# REGULARIZED LOSS FUNCTIONS
# ============================================================

def regularized_loss_fn(params, logical_0, logical_1, gamma, N_l, reg_lambda, reg_type="param"):
    """
    Regularized loss function for circuit optimization.

    Loss = (1 - Fe) + lambda * regularization

    Args:
        params: (N_depth, 4) circuit parameters
        logical_0, logical_1: CoherentKet GKP logical states
        gamma: loss parameter
        N_l: number of displacement terms (2^N_depth)
        reg_lambda: regularization strength
        reg_type: "param", "kraus", or "channel"

    Returns:
        loss: scalar loss value
    """
    alpha, beta = g(params, N_l)

    Fe = entanglement_fidelity_displacement(
        alpha, beta,
        logical_0.cs, logical_0.ds,
        logical_1.cs, logical_1.ds,
        gamma
    )

    # Compute regularization term
    if reg_type == "param":
        reg = param_norm_regularization(params)
    elif reg_type == "kraus":
        reg = kraus_identity_deviation(alpha, beta)
    elif reg_type == "channel":
        reg = channel_identity_overlap(alpha, beta)
    else:
        reg = 0.0

    return (1.0 - Fe) + reg_lambda * reg


# ============================================================
# CMA-ES OPTIMIZER WITH REGULARIZATION
# ============================================================

def optimize_regularized_cmaes(
    logical_0, logical_1, gamma,
    N_depth=6,
    reg_lambda=0.01,
    reg_type="param",
    popsize=80,
    maxiter=2000,
    sigma0=3.0,
    seed=42,
    verbose=True,
):
    """
    CMA-ES optimization with regularization toward identity.

    Args:
        logical_0, logical_1: CoherentKet GKP logical states
        gamma: loss parameter
        N_depth: CD+R circuit depth (N_l = 2^N_depth displacements)
        reg_lambda: regularization strength
        reg_type: "param", "kraus", or "channel"
        popsize: CMA-ES population size
        maxiter: maximum CMA-ES generations
        sigma0: initial step size
        seed: random seed
        verbose: print progress

    Returns:
        best_params: (N_depth, 4) optimized circuit parameters
        best_Fe: best entanglement fidelity (without regularization)
        info: dict with optimization details
    """
    import cma

    N_l = 2 ** N_depth
    n_params = N_depth * 4
    d_half = float(jnp.sqrt(jnp.pi / 2))

    def unpack(x_real):
        """Convert real parameter vector to complex (N_depth, 4) params."""
        p = jnp.zeros((N_depth, 4), dtype=jnp.complex64)
        for i in range(N_depth):
            p = p.at[i, 0].set(x_real[4*i] + 1j * x_real[4*i+1])
            p = p.at[i, 1].set(x_real[4*i+2])
            p = p.at[i, 2].set(x_real[4*i+3])
        return p

    @jax.jit
    def eval_Fe(p_complex):
        """Evaluate entanglement fidelity (without regularization)."""
        alpha, beta = g(p_complex, N_l)
        return entanglement_fidelity_displacement(
            alpha, beta,
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds, gamma)

    @jax.jit
    def eval_loss(p_complex):
        """Evaluate regularized loss."""
        return regularized_loss_fn(
            p_complex, logical_0, logical_1, gamma, N_l,
            reg_lambda, reg_type)

    # JIT warmup
    _ = eval_loss(jnp.zeros((N_depth, 4), dtype=jnp.complex64))
    _ = eval_Fe(jnp.zeros((N_depth, 4), dtype=jnp.complex64))

    def objective(x):
        return float(eval_loss(unpack(np.array(x))))

    # GKP-informed initial point
    x0 = np.zeros(n_params)
    x0[0] = d_half   # Re(d) for layer 0
    x0[3] = np.pi/2  # theta for layer 0 (balanced measurement)
    if N_depth > 1:
        x0[5] = d_half    # Re(d) for layer 1 (orthogonal direction)
        x0[7] = np.pi/2   # theta for layer 1

    Fe_id = float(eval_Fe(jnp.zeros((N_depth, 4), dtype=jnp.complex64)))

    if verbose:
        print(f"Regularized CMA-ES: N_depth={N_depth}, lambda={reg_lambda}, type={reg_type}")
        print(f"  Fe_id={Fe_id:.6f}")
        sys.stdout.flush()

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'maxiter': maxiter, 'popsize': popsize,
        'verbose': -1, 'seed': seed, 'tolfun': 1e-9,
    })

    gen = 0
    best_ever_Fe = 0.0
    t_start = time.time()

    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)

        # Track best Fe (not regularized loss)
        best_x_now = es.result.xbest
        Fe_now = float(eval_Fe(unpack(best_x_now)))
        best_ever_Fe = max(best_ever_Fe, Fe_now)

        if verbose and gen % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  gen {gen}: Fe={Fe_now:.6f} (ever={best_ever_Fe:.6f}) [{elapsed:.0f}s]")
            sys.stdout.flush()
        gen += 1

    best_params = unpack(es.result.xbest)
    best_Fe = float(eval_Fe(best_params))
    elapsed = time.time() - t_start

    if verbose:
        print(f"  Done ({elapsed:.0f}s, {gen} gens): Fe={best_Fe:.6f}")
        print(f"  Improvement over identity: {best_Fe - Fe_id:+.6f}")
        sys.stdout.flush()

    return best_params, best_Fe, {
        'Fe_id': Fe_id,
        'reg_lambda': reg_lambda,
        'reg_type': reg_type,
        'generations': gen,
        'elapsed': elapsed,
    }


# ============================================================
# MULTI-ROUND FIDELITY EVALUATION
# ============================================================

def evaluate_multi_round_fidelity(
    params, logical_0, logical_1, gamma, N_l, n_rounds=20, N=GKP_N
):
    """
    Evaluate fidelity after multiple rounds of loss + recovery.

    This tests whether the recovery channel is stable over repeated applications.

    Args:
        params: (N_depth, 4) circuit parameters
        logical_0, logical_1: CoherentKet GKP logical states
        gamma: loss parameter per round
        N_l: number of displacement terms
        n_rounds: number of loss+recovery cycles
        N: Fock space dimension

    Returns:
        fidelities: (n_rounds,) fidelity after each round
    """
    # Build Fock-basis operators
    alpha, beta = g(params, N_l)
    recovery_ops = channel_from_b(alpha, beta)
    loss_ops = make_pureloss_fock(gamma, rank=10, N=N)

    # Build Fock logical states
    fock_states = []
    for ck in [logical_0, logical_1]:
        coherents = jnp.squeeze(
            jax.vmap(lambda a: dqcoherent(N, a))(ck.ds)
        )
        if coherents.ndim == 3:
            coherents = coherents.squeeze(-1)
        psi = jnp.einsum('bn,b->n', coherents, ck.cs).reshape(-1, 1)
        psi = psi / jnp.sqrt(jnp.real(dag(psi) @ psi).squeeze())
        fock_states.append(psi)

    # Initialize maximally entangled state (in Choi form)
    # rho_{AB} = (1/2) sum_{i,j} |psi_i><psi_j| x |i><j|
    rho_list = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            rho_list[i][j] = fock_states[i] @ dag(fock_states[j])

    fidelities = []

    for round_idx in range(n_rounds):
        # Apply loss to each component
        for i in range(2):
            for j in range(2):
                rho_list[i][j] = apply_kraus_map_nonorm(loss_ops, rho_list[i][j])

        # Apply recovery to each component
        for i in range(2):
            for j in range(2):
                rho_list[i][j] = apply_kraus_map_nonorm(recovery_ops, rho_list[i][j])

        # Compute entanglement fidelity
        Fe = 0.0
        for i in range(2):
            for j in range(2):
                Fe += (dag(fock_states[i]) @ rho_list[i][j] @ fock_states[j]).squeeze()
        Fe = float(jnp.real(Fe) / 4.0)
        fidelities.append(Fe)

    return jnp.array(fidelities)


# ============================================================
# LAMBDA SWEEP
# ============================================================

def sweep_lambda_values(
    logical_0, logical_1, gamma,
    N_depth=6,
    lambda_values=None,
    reg_type="param",
    popsize=80,
    maxiter=1500,
    n_rounds_eval=20,
    verbose=True,
):
    """
    Sweep regularization strength to find optimal value.

    Args:
        logical_0, logical_1: CoherentKet GKP logical states
        gamma: loss parameter
        N_depth: circuit depth
        lambda_values: list of regularization strengths to try
        reg_type: regularization type
        popsize, maxiter: CMA-ES parameters
        n_rounds_eval: number of rounds for multi-round evaluation
        verbose: print progress

    Returns:
        results: dict with sweep results
    """
    if lambda_values is None:
        lambda_values = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

    N_l = 2 ** N_depth
    results = {
        'lambda_values': lambda_values,
        'single_round_Fe': [],
        'multi_round_Fe': [],
        'params': [],
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Lambda Sweep: gamma={gamma}, N_depth={N_depth}, reg_type={reg_type}")
        print(f"{'='*60}")
        sys.stdout.flush()

    for lam in lambda_values:
        if verbose:
            print(f"\n--- lambda={lam} ---")
            sys.stdout.flush()

        params, Fe_single, info = optimize_regularized_cmaes(
            logical_0, logical_1, gamma,
            N_depth=N_depth,
            reg_lambda=lam,
            reg_type=reg_type,
            popsize=popsize,
            maxiter=maxiter,
            verbose=verbose,
        )

        # Evaluate multi-round fidelity
        Fe_multi = evaluate_multi_round_fidelity(
            params, logical_0, logical_1, gamma, N_l, n_rounds=n_rounds_eval
        )

        results['single_round_Fe'].append(Fe_single)
        results['multi_round_Fe'].append(Fe_multi)
        results['params'].append(params)

        if verbose:
            print(f"  Single-round Fe: {Fe_single:.6f}")
            print(f"  Multi-round Fe (round {n_rounds_eval}): {float(Fe_multi[-1]):.6f}")
            sys.stdout.flush()

    # Find best lambda for multi-round stability
    final_Fe = [float(fe[-1]) for fe in results['multi_round_Fe']]
    best_idx = np.argmax(final_Fe)
    results['best_lambda'] = lambda_values[best_idx]
    results['best_multi_round_Fe'] = final_Fe[best_idx]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Best lambda={results['best_lambda']}: final Fe={results['best_multi_round_Fe']:.6f}")
        print(f"{'='*60}")
        sys.stdout.flush()

    return results


# ============================================================
# GAMMA SWEEP
# ============================================================

def compare_regularized_vs_unregularized(
    Delta=0.3,
    gamma_values=None,
    N_depth=6,
    best_lambda=0.01,
    reg_type="param",
    popsize=80,
    maxiter=1500,
    n_rounds_eval=20,
    verbose=True,
):
    """
    Compare regularized vs unregularized optimization across gamma values.

    Args:
        Delta: GKP envelope parameter
        gamma_values: loss parameters to test
        N_depth: circuit depth
        best_lambda: regularization strength for regularized runs
        reg_type: regularization type
        popsize, maxiter: CMA-ES parameters
        n_rounds_eval: number of rounds for multi-round evaluation
        verbose: print progress

    Returns:
        comparison: dict with comparison results
    """
    if gamma_values is None:
        gamma_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    N_l = 2 ** N_depth

    comparison = {
        'gamma_values': gamma_values,
        'unreg_single': [],
        'unreg_multi': [],
        'reg_single': [],
        'reg_multi': [],
        'Fe_id': [],
    }

    for gamma in gamma_values:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Gamma = {gamma}")
            print(f"{'='*60}")
            sys.stdout.flush()

        logical_0 = gkp_coherent_dm(mu=0, N_trunc=3, Delta=Delta, lattice='square')
        logical_1 = gkp_coherent_dm(mu=1, N_trunc=3, Delta=Delta, lattice='square')

        # Compute identity baseline
        Fe_id = float(entanglement_fidelity_displacement(
            jnp.ones((1, 1), dtype=jnp.complex64),
            jnp.zeros((1, 1), dtype=jnp.complex64),
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds, gamma))
        comparison['Fe_id'].append(Fe_id)

        # Unregularized
        if verbose:
            print("\n--- Unregularized ---")
        params_unreg, Fe_unreg, _ = optimize_regularized_cmaes(
            logical_0, logical_1, gamma,
            N_depth=N_depth,
            reg_lambda=0.0,
            reg_type=reg_type,
            popsize=popsize,
            maxiter=maxiter,
            verbose=verbose,
        )
        Fe_multi_unreg = evaluate_multi_round_fidelity(
            params_unreg, logical_0, logical_1, gamma, N_l, n_rounds=n_rounds_eval
        )
        comparison['unreg_single'].append(Fe_unreg)
        comparison['unreg_multi'].append(Fe_multi_unreg)

        # Regularized
        if verbose:
            print("\n--- Regularized ---")
        params_reg, Fe_reg, _ = optimize_regularized_cmaes(
            logical_0, logical_1, gamma,
            N_depth=N_depth,
            reg_lambda=best_lambda,
            reg_type=reg_type,
            popsize=popsize,
            maxiter=maxiter,
            verbose=verbose,
        )
        Fe_multi_reg = evaluate_multi_round_fidelity(
            params_reg, logical_0, logical_1, gamma, N_l, n_rounds=n_rounds_eval
        )
        comparison['reg_single'].append(Fe_reg)
        comparison['reg_multi'].append(Fe_multi_reg)

        if verbose:
            print(f"\n  Summary for gamma={gamma}:")
            print(f"    Fe_id:              {Fe_id:.6f}")
            print(f"    Unreg single:       {Fe_unreg:.6f}")
            print(f"    Unreg multi (r{n_rounds_eval}): {float(Fe_multi_unreg[-1]):.6f}")
            print(f"    Reg single:         {Fe_reg:.6f}")
            print(f"    Reg multi (r{n_rounds_eval}):   {float(Fe_multi_reg[-1]):.6f}")
            diff = float(Fe_multi_reg[-1]) - float(Fe_multi_unreg[-1])
            print(f"    Reg advantage:      {diff:+.6f}")
            sys.stdout.flush()

    return comparison


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Regularized Optimizer - Preventing Over-Correction")
    print("=" * 60)

    # Test parameters
    gamma = 0.15
    Delta = 0.3
    N_depth = 6

    logical_0 = gkp_coherent_dm(mu=0, N_trunc=3, Delta=Delta, lattice='square')
    logical_1 = gkp_coherent_dm(mu=1, N_trunc=3, Delta=Delta, lattice='square')

    # Compute baseline
    Fe_id = float(entanglement_fidelity_displacement(
        jnp.ones((1, 1), dtype=jnp.complex64),
        jnp.zeros((1, 1), dtype=jnp.complex64),
        logical_0.cs, logical_0.ds,
        logical_1.cs, logical_1.ds, gamma))
    print(f"\ngamma={gamma}, Delta={Delta}, Fe_id={Fe_id:.6f}")

    # ===== Phase 1: Lambda Sweep =====
    print("\n" + "=" * 60)
    print("Phase 1: Lambda Sweep")
    print("=" * 60)

    sweep_results = sweep_lambda_values(
        logical_0, logical_1, gamma,
        N_depth=N_depth,
        lambda_values=[0.0, 0.001, 0.005, 0.01, 0.02, 0.05],
        reg_type="param",
        popsize=60,
        maxiter=1000,
        n_rounds_eval=20,
        verbose=True,
    )

    print("\nSweep Summary:")
    print("-" * 50)
    print(f"{'Lambda':<10} {'Single Fe':<12} {'Multi Fe (r20)':<15}")
    print("-" * 50)
    for i, lam in enumerate(sweep_results['lambda_values']):
        single_fe = sweep_results['single_round_Fe'][i]
        multi_fe = float(sweep_results['multi_round_Fe'][i][-1])
        marker = " <-- best" if lam == sweep_results['best_lambda'] else ""
        print(f"{lam:<10.4f} {single_fe:<12.6f} {multi_fe:<15.6f}{marker}")

    # ===== Phase 2: Gamma Comparison =====
    print("\n" + "=" * 60)
    print("Phase 2: Regularized vs Unregularized Across Gamma")
    print("=" * 60)

    best_lambda = sweep_results['best_lambda']

    comparison = compare_regularized_vs_unregularized(
        Delta=Delta,
        gamma_values=[0.1, 0.15, 0.2],
        N_depth=N_depth,
        best_lambda=best_lambda,
        reg_type="param",
        popsize=60,
        maxiter=1000,
        n_rounds_eval=20,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Final Comparison Summary")
    print("=" * 60)
    print(f"{'Gamma':<8} {'Fe_id':<8} {'Unreg(1)':<10} {'Unreg(20)':<10} {'Reg(1)':<10} {'Reg(20)':<10} {'Advantage':<10}")
    print("-" * 66)
    for i, gamma in enumerate(comparison['gamma_values']):
        fe_id = comparison['Fe_id'][i]
        unreg1 = comparison['unreg_single'][i]
        unreg20 = float(comparison['unreg_multi'][i][-1])
        reg1 = comparison['reg_single'][i]
        reg20 = float(comparison['reg_multi'][i][-1])
        advantage = reg20 - unreg20
        print(f"{gamma:<8.2f} {fe_id:<8.4f} {unreg1:<10.4f} {unreg20:<10.4f} {reg1:<10.4f} {reg20:<10.4f} {advantage:+10.4f}")
