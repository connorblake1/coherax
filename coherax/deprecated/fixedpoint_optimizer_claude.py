"""
fixedpoint_optimizer_claude.py

Fixed-point stability optimizer for GKP recovery.

Instead of optimizing one-shot entanglement fidelity, this optimizes for
multi-round stability: finding R such that (R ∘ L)^N maintains high fidelity.

The key insight is that a recovery optimized for one-shot Fe may degrade
the state over many rounds. A fixed-point approach finds R where the
combined channel R∘L has the GKP code space as an approximate fixed point.

Methods implemented:
  1. Multi-round objective: maximize Fe after N rounds of (loss → recovery)
  2. Geometric mean fidelity: maximize (Fe_1 * Fe_2 * ... * Fe_N)^(1/N)
  3. Minimum fidelity: maximize min(Fe_1, Fe_2, ..., Fe_N)
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from functools import partial
from typing import Tuple, Dict, Any

from coherax.characteristic_jax_utils import (
    CoherentKet,
    coherent_overlap,
    aOmegab,
    g,
    gkp_coherent_dm,
    make_pureloss_fock,
    apply_kraus_map_nonorm,
    dqcoherent,
    dag,
    channel_from_b,
    GKP_N,
    dqdisplace,
)


# ============================================================
# HELPER: channel_from_b with configurable N
# ============================================================

def channel_from_b_N(alphas: jnp.ndarray, betas: jnp.ndarray, N: int):
    """
    Build Kraus operators from displacement coefficients with configurable Fock dimension.

    K_j = sum_i alpha[j,i] * D(beta[j,i])

    Args:
        alphas: (n_kraus, N_disp) complex coefficients
        betas: (n_kraus, N_disp) complex displacements
        N: Fock space dimension

    Returns:
        ops: (n_kraus, N, N) Kraus operators
    """
    n_kraus = alphas.shape[0]
    N_disp = alphas.shape[1]

    ops = jnp.zeros((n_kraus, N, N), dtype=jnp.complex64)

    for j in range(n_kraus):
        op_j = jnp.zeros((N, N), dtype=jnp.complex64)
        for i in range(N_disp):
            op_j = op_j + alphas[j, i] * dqdisplace(N, betas[j, i]).astype(jnp.complex64)
        ops = ops.at[j].set(op_j)

    return ops


# ============================================================
# FOCK-BASIS MULTI-ROUND
# ============================================================

def multiround_fidelity_fock(
    params: jnp.ndarray,
    logical_0: CoherentKet,
    logical_1: CoherentKet,
    gamma: float,
    n_rounds: int = 5,
    N: int = None,
    loss_rank: int = 10,
) -> float:
    """
    Compute entanglement fidelity after n_rounds of (loss -> recovery) in Fock basis.

    This is the accurate calculation using full density matrix evolution.

    Args:
        params: (N_depth, 4) circuit parameters
        logical_0, logical_1: GKP logical states
        gamma: loss parameter
        n_rounds: number of (loss -> recovery) cycles
        N: Fock space dimension
        loss_rank: number of loss Kraus operators

    Returns:
        Fe: entanglement fidelity after n_rounds
    """
    if N is None:
        N = GKP_N
    N_depth = params.shape[0]
    N_l = 2 ** N_depth

    # Build recovery Kraus operators in Fock basis
    alpha, beta = g(params, N_l)
    recovery_ops = channel_from_b_N(alpha, beta, N)

    # Build loss Kraus operators
    loss_ops = make_pureloss_fock(gamma, rank=loss_rank, N=N)

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

    # Compute entanglement fidelity
    Fe = 0.0
    for mu in range(2):
        for nu in range(2):
            # Initial density matrix element
            rho = fock_states[mu] @ dag(fock_states[nu])

            # Apply n_rounds of (loss -> recovery)
            for _ in range(n_rounds):
                rho = apply_kraus_map_nonorm(loss_ops, rho)
                rho = apply_kraus_map_nonorm(recovery_ops, rho)

            # Final fidelity contribution
            Fe += jnp.real(dag(fock_states[mu]) @ rho @ fock_states[nu]).squeeze()

    return float(Fe / 4.0)


# ============================================================
# FIXED-POINT CMA-ES OPTIMIZER
# ============================================================

def fixedpoint_cmaes(
    logical_0: CoherentKet,
    logical_1: CoherentKet,
    gamma: float,
    n_rounds: int = 5,
    N_depth: int = 6,
    popsize: int = 80,
    maxiter: int = 1500,
    sigma0: float = 3.0,
    seed: int = 42,
    objective: str = 'multiround',  # 'multiround', 'geometric', 'minimum'
    verbose: bool = True,
) -> Tuple[jnp.ndarray, float, Dict[str, Any]]:
    """
    CMA-ES optimization for fixed-point stability.

    Instead of maximizing one-shot Fe, maximizes stability over multiple rounds.

    Objective options:
      - 'multiround': Maximize Fe after n_rounds
      - 'geometric': Maximize geometric mean of per-round fidelities
      - 'minimum': Maximize minimum fidelity across rounds (minimax)

    Args:
        logical_0, logical_1: GKP logical states
        gamma: loss parameter
        n_rounds: number of rounds to optimize for
        N_depth: circuit depth
        popsize: CMA-ES population
        maxiter: max generations
        sigma0: initial step size
        seed: random seed
        objective: optimization objective type
        verbose: print progress

    Returns:
        best_params: optimized (N_depth, 4) parameters
        best_value: best objective value achieved
        info: optimization details
    """
    import sys
    import cma
    import time

    N_l = 2 ** N_depth
    n_params = N_depth * 4
    d_half = float(jnp.sqrt(jnp.pi / 2))

    # Fock basis dimensions
    N = min(GKP_N, 80)  # Limit for speed
    loss_rank = min(10, N // 2)

    def unpack(x_real):
        p = jnp.zeros((N_depth, 4), dtype=jnp.complex64)
        for i in range(N_depth):
            p = p.at[i, 0].set(x_real[4*i] + 1j * x_real[4*i+1])
            p = p.at[i, 1].set(x_real[4*i+2])
            p = p.at[i, 2].set(x_real[4*i+3])
        return p

    # Build Fock operators once
    loss_ops = make_pureloss_fock(gamma, rank=loss_rank, N=N)

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

    @jax.jit
    def compute_per_round_fe(recovery_ops):
        """Compute Fe after each round."""
        fe_per_round = []

        # Initial state for tracking
        rho_list = []
        for mu in range(2):
            for nu in range(2):
                rho_list.append(fock_states[mu] @ dag(fock_states[nu]))

        for r_idx in range(n_rounds):
            # Apply (loss -> recovery) to all rho elements
            new_rho_list = []
            for rho in rho_list:
                rho = apply_kraus_map_nonorm(loss_ops, rho)
                rho = apply_kraus_map_nonorm(recovery_ops, rho)
                new_rho_list.append(rho)
            rho_list = new_rho_list

            # Compute Fe for this round
            Fe = 0.0
            idx = 0
            for mu in range(2):
                for nu in range(2):
                    Fe += jnp.real(dag(fock_states[mu]) @ rho_list[idx] @ fock_states[nu]).squeeze()
                    idx += 1
            fe_per_round.append(Fe / 4.0)

        return jnp.array(fe_per_round)

    def objective_fn(x):
        params = unpack(np.array(x))
        alpha, beta = g(params, N_l)
        recovery_ops = channel_from_b_N(alpha, beta, N)

        fe_rounds = compute_per_round_fe(recovery_ops)

        if objective == 'multiround':
            # Maximize final round fidelity
            return -float(fe_rounds[-1])
        elif objective == 'geometric':
            # Maximize geometric mean
            return -float(jnp.exp(jnp.mean(jnp.log(jnp.maximum(fe_rounds, 1e-10)))))
        elif objective == 'minimum':
            # Maximize minimum (worst-round)
            return -float(jnp.min(fe_rounds))
        else:
            raise ValueError(f"Unknown objective: {objective}")

    # Warm-up JIT
    _ = objective_fn(np.zeros(n_params))

    # Initial point
    x0 = np.zeros(n_params)
    x0[0] = d_half
    x0[3] = np.pi/2
    if N_depth > 1:
        x0[5] = d_half
        x0[7] = np.pi/2

    Fe_id = -objective_fn(np.zeros(n_params))

    if verbose:
        print(f"Fixed-Point CMA-ES: N_depth={N_depth}, n_rounds={n_rounds}")
        print(f"  Objective: {objective}")
        print(f"  Baseline (identity): {Fe_id:.6f}")
        sys.stdout.flush()

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'maxiter': maxiter,
        'popsize': popsize,
        'verbose': -1,
        'seed': seed,
        'tolfun': 1e-9,
    })

    gen = 0
    best_ever = Fe_id
    t_start = time.time()

    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective_fn(x) for x in solutions]
        es.tell(solutions, fitnesses)

        best_now = -es.result.fbest
        if best_now > best_ever:
            best_ever = best_now

        if verbose and gen % 50 == 0:
            elapsed = time.time() - t_start
            print(f"  gen {gen}: best={best_now:.6f} (ever={best_ever:.6f}) [{elapsed:.0f}s]")
            sys.stdout.flush()
        gen += 1

    best_value = -es.result.fbest
    best_params = unpack(es.result.xbest)
    elapsed = time.time() - t_start

    if verbose:
        print(f"  Done ({elapsed:.0f}s, {gen} gens)")
        print(f"  Best {objective} = {best_value:.6f}")
        print(f"  Improvement over identity: {best_value - Fe_id:+.6f}")
        sys.stdout.flush()

    return best_params, best_value, {
        'Fe_id': Fe_id,
        'objective': objective,
        'n_rounds': n_rounds,
        'generations': gen,
        'elapsed': elapsed,
    }


def fixedpoint_bipop(
    logical_0: CoherentKet,
    logical_1: CoherentKet,
    gamma: float,
    n_rounds: int = 5,
    N_depth: int = 6,
    n_restarts: int = 5,
    popsize: int = 80,
    maxiter: int = 1000,
    sigma0: float = 3.0,
    objective: str = 'multiround',
    verbose: bool = True,
) -> Tuple[jnp.ndarray, float, Dict[str, Any]]:
    """
    BIPOP-style multi-restart CMA-ES for fixed-point optimization.
    """
    import sys
    import time

    if verbose:
        print(f"\n{'='*60}")
        print(f"Fixed-Point BIPOP CMA-ES")
        print(f"  gamma={gamma}, n_rounds={n_rounds}, N_depth={N_depth}")
        print(f"  objective={objective}, restarts={n_restarts}")
        print(f"{'='*60}")
        sys.stdout.flush()

    best_params = None
    best_value = 0.0
    all_trials = []
    t_total = time.time()

    for trial in range(n_restarts):
        if verbose:
            print(f"\n--- Trial {trial} ---")
            sys.stdout.flush()

        params, value, info = fixedpoint_cmaes(
            logical_0, logical_1, gamma,
            n_rounds=n_rounds,
            N_depth=N_depth,
            popsize=popsize,
            maxiter=maxiter,
            sigma0=sigma0,
            seed=trial,
            objective=objective,
            verbose=verbose,
        )

        all_trials.append({'seed': trial, 'value': value, **info})

        if value > best_value:
            best_value = value
            best_params = params
            if verbose:
                print(f"  >> New best! value={best_value:.6f}")
                sys.stdout.flush()

    elapsed_total = time.time() - t_total

    if verbose:
        print(f"\n{'='*60}")
        print(f"BIPOP Summary")
        print(f"  Best {objective} = {best_value:.6f}")
        print(f"  Total time: {elapsed_total:.0f}s")
        print(f"{'='*60}")
        sys.stdout.flush()

    return best_params, best_value, {
        'trials': all_trials,
        'objective': objective,
        'n_rounds': n_rounds,
        'total_time': elapsed_total,
    }


# ============================================================
# COMPARISON UTILITIES
# ============================================================

def compare_oneshot_vs_fixedpoint(
    logical_0: CoherentKet,
    logical_1: CoherentKet,
    gamma: float,
    n_rounds: int = 10,
    N_depth: int = 6,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compare one-shot optimized recovery vs fixed-point optimized recovery.

    Runs both optimization approaches and evaluates their multi-round performance.
    """
    import sys
    from coherax.coherent_tree_optimizer_claude import optimize_cmaes_flat

    if verbose:
        print(f"\n{'='*60}")
        print(f"Comparison: One-Shot vs Fixed-Point Optimization")
        print(f"  gamma={gamma}, n_rounds={n_rounds}, N_depth={N_depth}")
        print(f"{'='*60}")
        sys.stdout.flush()

    # One-shot optimization
    if verbose:
        print("\n--- One-Shot Optimization ---")
        sys.stdout.flush()

    params_oneshot, Fe_oneshot, info_oneshot = optimize_cmaes_flat(
        logical_0, logical_1, gamma,
        N_depth=N_depth,
        popsize=80,
        maxiter=1500,
        verbose=verbose,
    )

    # Fixed-point optimization
    if verbose:
        print("\n--- Fixed-Point Optimization ---")
        sys.stdout.flush()

    params_fp, Fe_fp, info_fp = fixedpoint_cmaes(
        logical_0, logical_1, gamma,
        n_rounds=n_rounds,
        N_depth=N_depth,
        popsize=80,
        maxiter=1500,
        objective='multiround',
        verbose=verbose,
    )

    # Evaluate both over multiple rounds
    if verbose:
        print("\n--- Multi-Round Evaluation ---")
        sys.stdout.flush()

    fe_oneshot_rounds = []
    fe_fp_rounds = []

    N = min(GKP_N, 80)

    for r in range(1, n_rounds + 1):
        fe_os = multiround_fidelity_fock(params_oneshot, logical_0, logical_1, gamma, n_rounds=r, N=N)
        fe_fp = multiround_fidelity_fock(params_fp, logical_0, logical_1, gamma, n_rounds=r, N=N)
        fe_oneshot_rounds.append(fe_os)
        fe_fp_rounds.append(fe_fp)

        if verbose:
            print(f"  Round {r}: one-shot={fe_os:.6f}, fixed-point={fe_fp:.6f}, "
                  f"diff={fe_fp-fe_os:+.6f}")
            sys.stdout.flush()

    if verbose:
        print(f"\n--- Summary ---")
        print(f"  One-shot final Fe (round {n_rounds}): {fe_oneshot_rounds[-1]:.6f}")
        print(f"  Fixed-point final Fe (round {n_rounds}): {fe_fp_rounds[-1]:.6f}")
        print(f"  Improvement: {fe_fp_rounds[-1] - fe_oneshot_rounds[-1]:+.6f}")
        sys.stdout.flush()

    return {
        'gamma': gamma,
        'n_rounds': n_rounds,
        'params_oneshot': params_oneshot,
        'params_fixedpoint': params_fp,
        'fe_oneshot_rounds': fe_oneshot_rounds,
        'fe_fixedpoint_rounds': fe_fp_rounds,
        'improvement_final': fe_fp_rounds[-1] - fe_oneshot_rounds[-1],
    }


# ============================================================
# CODE SPACE FIXED POINT OPTIMIZER
# ============================================================

def compute_code_space_channel(
    recovery_ops: jnp.ndarray,
    loss_ops: jnp.ndarray,
    psi_0: jnp.ndarray,
    psi_1: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the 4x4 Liouville representation of R∘L restricted to code space.

    The code space is spanned by |0_L⟩, |1_L⟩. A general code state is:
        ρ = Σ_{μ,ν} c_{μν} |μ⟩⟨ν|

    The channel (R∘L) maps this to another code-space state. We compute
    the 4x4 superoperator matrix S where:
        vec((R∘L)(ρ)) = S @ vec(ρ)

    For the code space to be a fixed point, S should be identity.

    Returns:
        S: (4, 4) complex superoperator matrix
    """
    # Basis for code space density matrices: |0⟩⟨0|, |0⟩⟨1|, |1⟩⟨0|, |1⟩⟨1|
    basis_dm = [
        psi_0 @ dag(psi_0),  # |0⟩⟨0|
        psi_0 @ dag(psi_1),  # |0⟩⟨1|
        psi_1 @ dag(psi_0),  # |1⟩⟨0|
        psi_1 @ dag(psi_1),  # |1⟩⟨1|
    ]

    # Apply R∘L to each basis element
    output_dm = []
    for rho in basis_dm:
        rho_after_loss = apply_kraus_map_nonorm(loss_ops, rho)
        rho_after_recovery = apply_kraus_map_nonorm(recovery_ops, rho_after_loss)
        output_dm.append(rho_after_recovery)

    # Project outputs back to code space to get S matrix
    # S[i,j] = ⟨basis_i | (R∘L)(basis_j) ⟩ in the Hilbert-Schmidt inner product
    # But for Liouville rep, we need: output_vec = S @ input_vec
    # where vec(ρ) = [⟨0|ρ|0⟩, ⟨0|ρ|1⟩, ⟨1|ρ|0⟩, ⟨1|ρ|1⟩]

    S = jnp.zeros((4, 4), dtype=jnp.complex64)
    for j, rho_out in enumerate(output_dm):
        # Extract code-space components of output
        S = S.at[0, j].set((dag(psi_0) @ rho_out @ psi_0).squeeze())
        S = S.at[1, j].set((dag(psi_0) @ rho_out @ psi_1).squeeze())
        S = S.at[2, j].set((dag(psi_1) @ rho_out @ psi_0).squeeze())
        S = S.at[3, j].set((dag(psi_1) @ rho_out @ psi_1).squeeze())

    return S


def code_space_fixed_point_loss(S: jnp.ndarray) -> jnp.ndarray:
    """
    Compute loss measuring deviation from identity channel on code space.

    For a true fixed point, S should equal I_4.

    Returns:
        loss: ||S - I||_F^2
    """
    I4 = jnp.eye(4, dtype=jnp.complex64)
    return jnp.sum(jnp.abs(S - I4)**2)


def bloch_sphere_fidelities(
    recovery_ops: jnp.ndarray,
    loss_ops: jnp.ndarray,
    psi_0: jnp.ndarray,
    psi_1: jnp.ndarray,
    n_points: int = 8,
) -> jnp.ndarray:
    """
    Compute fidelities for states sampled around the Bloch sphere.

    Tests |0⟩, |1⟩, |+⟩, |-⟩, |+i⟩, |-i⟩, and additional samples.

    Returns:
        fidelities: (n_points,) array of fidelities F(ρ, (R∘L)(ρ))
    """
    # Key Bloch sphere points
    angles = [(0, 0), (jnp.pi, 0),  # |0⟩, |1⟩
              (jnp.pi/2, 0), (jnp.pi/2, jnp.pi),  # |+⟩, |-⟩
              (jnp.pi/2, jnp.pi/2), (jnp.pi/2, 3*jnp.pi/2)]  # |+i⟩, |-i⟩

    # Add more points if requested
    if n_points > 6:
        for i in range(n_points - 6):
            theta = jnp.pi * (i + 1) / (n_points - 5)
            phi = 2 * jnp.pi * i / (n_points - 6 + 1)
            angles.append((theta, phi))

    fidelities = []
    for theta, phi in angles[:n_points]:
        # |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ}sin(θ/2)|1⟩
        c0 = jnp.cos(theta / 2)
        c1 = jnp.exp(1j * phi) * jnp.sin(theta / 2)
        psi = c0 * psi_0 + c1 * psi_1
        psi = psi / jnp.sqrt(jnp.real(dag(psi) @ psi).squeeze())

        rho = psi @ dag(psi)
        rho_out = apply_kraus_map_nonorm(loss_ops, rho)
        rho_out = apply_kraus_map_nonorm(recovery_ops, rho_out)

        # Fidelity F(ρ, σ) = Tr(ρσ) for pure ρ
        fid = jnp.real(dag(psi) @ rho_out @ psi).squeeze()
        fidelities.append(fid)

    return jnp.array(fidelities)


def true_fixedpoint_cmaes(
    logical_0: CoherentKet,
    logical_1: CoherentKet,
    gamma: float,
    N_depth: int = 6,
    popsize: int = 80,
    maxiter: int = 2000,
    sigma0: float = 3.0,
    seed: int = 42,
    init_params: jnp.ndarray = None,
    objective: str = 'channel',  # 'channel', 'bloch_min', 'bloch_avg'
    verbose: bool = True,
) -> Tuple[jnp.ndarray, float, Dict[str, Any]]:
    """
    CMA-ES optimization for true code-space fixed point.

    Instead of just maximizing entanglement fidelity, this ensures that
    ALL logical states are preserved by (R∘L).

    Objective options:
      - 'channel': Minimize ||S - I||_F where S is the code-space superoperator
      - 'bloch_min': Maximize minimum fidelity over Bloch sphere samples
      - 'bloch_avg': Maximize average fidelity over Bloch sphere samples

    Args:
        logical_0, logical_1: GKP logical states
        gamma: loss parameter
        N_depth: circuit depth
        popsize: CMA-ES population
        maxiter: max generations
        sigma0: initial step size
        seed: random seed
        init_params: optional (N_depth, 4) initial parameters (e.g., from one-shot)
        objective: optimization objective type
        verbose: print progress

    Returns:
        best_params: optimized (N_depth, 4) parameters
        best_value: best objective value achieved
        info: optimization details
    """
    import sys
    import cma
    import time

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

    def pack(p_complex):
        x = np.zeros(n_params)
        for i in range(N_depth):
            x[4*i] = float(p_complex[i, 0].real)
            x[4*i+1] = float(p_complex[i, 0].imag)
            x[4*i+2] = float(p_complex[i, 1].real)
            x[4*i+3] = float(p_complex[i, 2].real)
        return x

    # Build Fock operators once
    loss_ops = make_pureloss_fock(gamma, rank=loss_rank, N=N)

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

    psi_0, psi_1 = fock_states

    def objective_fn(x):
        params = unpack(np.array(x))
        alpha, beta = g(params, N_l)
        recovery_ops = channel_from_b_N(alpha, beta, N)

        if objective == 'channel':
            # Minimize ||S - I||_F
            S = compute_code_space_channel(recovery_ops, loss_ops, psi_0, psi_1)
            loss = code_space_fixed_point_loss(S)
            return float(loss)

        elif objective == 'bloch_min':
            # Maximize minimum Bloch sphere fidelity
            fids = bloch_sphere_fidelities(recovery_ops, loss_ops, psi_0, psi_1, n_points=8)
            return -float(jnp.min(fids))

        elif objective == 'bloch_avg':
            # Maximize average Bloch sphere fidelity (= entanglement fidelity)
            fids = bloch_sphere_fidelities(recovery_ops, loss_ops, psi_0, psi_1, n_points=8)
            return -float(jnp.mean(fids))

        else:
            raise ValueError(f"Unknown objective: {objective}")

    # Warm-up JIT
    _ = objective_fn(np.zeros(n_params))

    # Initial point
    if init_params is not None:
        x0 = pack(init_params)
        if verbose:
            print(f"  Starting from provided initial params")
    else:
        x0 = np.zeros(n_params)
        x0[0] = d_half
        x0[3] = np.pi/2
        if N_depth > 1:
            x0[5] = d_half
            x0[7] = np.pi/2

    baseline = objective_fn(np.zeros(n_params))
    init_val = objective_fn(x0)

    if verbose:
        print(f"True Fixed-Point CMA-ES: N_depth={N_depth}, gamma={gamma}")
        print(f"  Objective: {objective}")
        if objective == 'channel':
            print(f"  Baseline (identity): ||S-I||_F = {baseline:.6f}")
            print(f"  Initial: ||S-I||_F = {init_val:.6f}")
        else:
            print(f"  Baseline (identity): min_fid = {-baseline:.6f}")
            print(f"  Initial: min_fid = {-init_val:.6f}")
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

        if verbose and gen % 100 == 0:
            elapsed = time.time() - t_start
            if objective == 'channel':
                print(f"  gen {gen}: ||S-I||_F = {best_now:.6f} (ever={best_ever:.6f}) [{elapsed:.0f}s]")
            else:
                print(f"  gen {gen}: min_fid = {-best_now:.6f} (ever={-best_ever:.6f}) [{elapsed:.0f}s]")
            sys.stdout.flush()
        gen += 1

    best_value = es.result.fbest
    best_params = unpack(es.result.xbest)
    elapsed = time.time() - t_start

    # Compute final diagnostics
    alpha, beta = g(best_params, N_l)
    recovery_ops = channel_from_b_N(alpha, beta, N)
    S_final = compute_code_space_channel(recovery_ops, loss_ops, psi_0, psi_1)
    fids_final = bloch_sphere_fidelities(recovery_ops, loss_ops, psi_0, psi_1, n_points=6)

    if verbose:
        print(f"  Done ({elapsed:.0f}s, {gen} gens)")
        print(f"  Final ||S-I||_F = {float(code_space_fixed_point_loss(S_final)):.6f}")
        print(f"  Bloch sphere fidelities:")
        labels = ['|0⟩', '|1⟩', '|+⟩', '|-⟩', '|+i⟩', '|-i⟩']
        for i, (lbl, fid) in enumerate(zip(labels, fids_final)):
            print(f"    {lbl}: {float(fid):.6f}")
        print(f"  Min fidelity: {float(jnp.min(fids_final)):.6f}")
        print(f"  Entanglement fidelity: {float(jnp.mean(fids_final[:2])):.6f}")
        sys.stdout.flush()

    return best_params, best_value, {
        'objective': objective,
        'S_final': S_final,
        'bloch_fidelities': fids_final,
        'generations': gen,
        'elapsed': elapsed,
    }


def modify_oneshot_to_fixedpoint(
    logical_0: CoherentKet,
    logical_1: CoherentKet,
    gamma: float,
    N_depth: int = 6,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Take a one-shot optimized solution and modify it to be a true fixed point.

    1. First run one-shot CMA-ES to get a good starting point
    2. Then fine-tune with the code-space fixed-point objective

    Returns comparison of both approaches.
    """
    import sys
    from coherax.coherent_tree_optimizer_claude import optimize_cmaes_flat

    if verbose:
        print(f"\n{'='*60}")
        print(f"Modifying One-Shot to True Fixed Point")
        print(f"  gamma={gamma}, N_depth={N_depth}")
        print(f"{'='*60}")
        sys.stdout.flush()

    # Step 1: One-shot optimization
    if verbose:
        print("\n--- Step 1: One-Shot CMA-ES ---")
        sys.stdout.flush()

    params_oneshot, Fe_oneshot, info_oneshot = optimize_cmaes_flat(
        logical_0, logical_1, gamma,
        N_depth=N_depth,
        popsize=80,
        maxiter=1500,
        verbose=verbose,
    )

    # Step 2: Fixed-point fine-tuning starting from one-shot
    if verbose:
        print("\n--- Step 2: Fixed-Point Fine-Tuning ---")
        sys.stdout.flush()

    params_fp, loss_fp, info_fp = true_fixedpoint_cmaes(
        logical_0, logical_1, gamma,
        N_depth=N_depth,
        popsize=80,
        maxiter=2000,
        sigma0=0.5,  # Smaller step size for fine-tuning
        init_params=params_oneshot,
        objective='channel',
        verbose=verbose,
    )

    # Step 3: Also try bloch_min objective
    if verbose:
        print("\n--- Step 3: Bloch-Min Fine-Tuning ---")
        sys.stdout.flush()

    params_bloch, loss_bloch, info_bloch = true_fixedpoint_cmaes(
        logical_0, logical_1, gamma,
        N_depth=N_depth,
        popsize=80,
        maxiter=2000,
        sigma0=0.5,
        init_params=params_oneshot,
        objective='bloch_min',
        verbose=verbose,
    )

    # Compare multi-round performance
    if verbose:
        print("\n--- Multi-Round Comparison ---")
        sys.stdout.flush()

    N = min(GKP_N, 80)

    results = {
        'oneshot': {'params': params_oneshot, 'rounds': []},
        'channel_fp': {'params': params_fp, 'rounds': []},
        'bloch_fp': {'params': params_bloch, 'rounds': []},
    }

    for name, data in results.items():
        for r in [1, 2, 3, 5, 10]:
            fe = multiround_fidelity_fock(
                data['params'], logical_0, logical_1, gamma,
                n_rounds=r, N=N
            )
            data['rounds'].append((r, fe))

    if verbose:
        print(f"\n  {'Round':<8} {'One-Shot':<12} {'Channel-FP':<12} {'Bloch-FP':<12}")
        print(f"  {'-'*44}")
        for i, r in enumerate([1, 2, 3, 5, 10]):
            os_fe = results['oneshot']['rounds'][i][1]
            ch_fe = results['channel_fp']['rounds'][i][1]
            bl_fe = results['bloch_fp']['rounds'][i][1]
            print(f"  {r:<8} {os_fe:<12.6f} {ch_fe:<12.6f} {bl_fe:<12.6f}")
        sys.stdout.flush()

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("True Code-Space Fixed Point Optimizer")
    print("=" * 60)

    # Test parameters
    gamma = 0.05
    N_depth = 6

    # Build GKP states
    logical_0 = gkp_coherent_dm(mu=0, N_trunc=3, Delta=0.3, lattice='square')
    logical_1 = gkp_coherent_dm(mu=1, N_trunc=3, Delta=0.3, lattice='square')

    print(f"\ngamma={gamma}, N_depth={N_depth}")
    print(f"GKP: Delta=0.3, N_trunc=3")
    sys.stdout.flush()

    # Run the modification pipeline
    results = modify_oneshot_to_fixedpoint(
        logical_0, logical_1, gamma,
        N_depth=N_depth,
        verbose=True,
    )

    # Save results
    np.savez(
        'results/true_fixedpoint_comparison.npz',
        gamma=gamma,
        params_oneshot=np.array(results['oneshot']['params']),
        params_channel_fp=np.array(results['channel_fp']['params']),
        params_bloch_fp=np.array(results['bloch_fp']['params']),
        rounds_oneshot=np.array(results['oneshot']['rounds']),
        rounds_channel_fp=np.array(results['channel_fp']['rounds']),
        rounds_bloch_fp=np.array(results['bloch_fp']['rounds']),
    )
    print("\nResults saved to results/true_fixedpoint_comparison.npz")
