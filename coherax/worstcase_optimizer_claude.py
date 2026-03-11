"""
worstcase_optimizer_claude.py

Worst-case fidelity optimization for GKP error recovery.

Instead of optimizing for average entanglement fidelity over the codespace,
this module optimizes for the worst-case fidelity to ensure no codespace
state accumulates errors faster than others.

Key idea: Sample states on the Bloch sphere of the logical qubit codespace
|psi(theta, phi)> = cos(theta/2)|0_L> + e^{i*phi}sin(theta/2)|1_L>
and optimize for the minimum fidelity over all such states.

This is more robust than average-case optimization for multi-round error
correction, where the worst-case state determines the error threshold.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from functools import partial
import sys
import time

from coherax import (
    CoherentKet,
    coherent_overlap,
    aOmegab,
    g,
    gkp_coherent_dm,
    make_pureloss_fock,
    apply_kraus_map_nonorm,
    GKP_N,
    dqcoherent,
    channel_from_b,
    dag,
)


# ============================================================
# WORST-CASE FIDELITY COMPUTATION (COHERENT BASIS)
# ============================================================

@jax.jit
def state_fidelity_single_coherent(
    alpha, beta, c_psi, d_psi, gamma
):
    """
    Compute state fidelity for a single pure state |psi> after loss + recovery.

    F = <psi| R(E(|psi><psi|)) |psi>

    where |psi> = sum_a c_psi[a] |d_psi[a]> and R is parameterized by (alpha, beta).

    Args:
        alpha: (n_kraus, N_disp) complex Kraus coefficients
        beta: (n_kraus, N_disp) complex displacement positions
        c_psi: (A,) complex coefficients for |psi>
        d_psi: (A,) complex positions for |psi>
        gamma: loss parameter

    Returns:
        F: state fidelity (real scalar)
    """
    t = jnp.sqrt(1.0 - gamma)
    r = jnp.sqrt(gamma)
    n_kraus = alpha.shape[0]
    A = c_psi.shape[0]

    # Environment overlap: <r*d_b | r*d_a> for input state
    env_ov = coherent_overlap(
        r * d_psi.reshape(-1, 1),   # (A, 1)
        r * d_psi.reshape(1, -1),   # (1, A)
    )  # (A, A)

    F = 0.0 + 0j

    for k in range(n_kraus):
        # L_k[a] = <psi| R_k |t*d_psi[a]>
        # where R_k = sum_j alpha[k,j] D(beta[k,j])

        td = t * d_psi  # (A,)

        # phase[j,a] = exp(-i * aOmegab(beta[k,j], t*d_psi[a]))
        phase = jnp.exp(-1j * aOmegab(
            beta[k, :, None],    # (N_disp, 1)
            td[None, :],         # (1, A)
        ))  # (N_disp, A)

        # shifted[j,a] = beta[k,j] + t*d_psi[a]
        shifted = beta[k, :, None] + td[None, :]  # (N_disp, A)

        # ovlp[p,j,a] = <d_psi[p] | shifted[j,a]>
        ovlp = coherent_overlap(
            d_psi[:, None, None],   # (A, 1, 1)
            shifted[None, :, :],    # (1, N_disp, A)
        )  # (A, N_disp, A)

        # L_k[a] = sum_{p,j} conj(c_psi[p]) * alpha[k,j] * phase[j,a] * ovlp[p,j,a]
        L_k = jnp.einsum(
            'p,j,ja,pja->a',
            jnp.conj(c_psi), alpha[k], phase, ovlp,
        )  # (A,)

        # Contribution: sum_{a,b} c_psi[a] * conj(c_psi[b]) * L_k[a] * conj(L_k[b]) * env_ov[b,a]
        # = (c_psi * L_k)^dag @ env_ov @ (c_psi * L_k)
        v = c_psi * L_k  # (A,)
        F += jnp.conj(v) @ env_ov @ v

    return jnp.real(F)


def bloch_state_coeffs(theta, phi, c_0, d_0, c_1, d_1):
    """
    Construct Bloch sphere state coefficients.

    |psi(theta, phi)> = cos(theta/2)|0_L> + e^{i*phi}sin(theta/2)|1_L>

    Returns:
        c_psi: (A0+A1,) combined coefficients
        d_psi: (A0+A1,) combined positions
    """
    A0 = c_0.shape[0]
    A1 = c_1.shape[0]

    coeff_0 = jnp.cos(theta / 2)
    coeff_1 = jnp.sin(theta / 2) * jnp.exp(1j * phi)

    c_psi = jnp.concatenate([coeff_0 * c_0, coeff_1 * c_1])
    d_psi = jnp.concatenate([d_0, d_1])

    return c_psi, d_psi


@partial(jax.jit, static_argnums=(5, 6))
def worst_case_fidelity_grid(
    alpha, beta, c_0, d_0, c_1, d_1, gamma, n_theta=16, n_phi=32
):
    """
    Compute worst-case fidelity by grid search over Bloch sphere.

    Samples (theta, phi) uniformly and returns the minimum fidelity.

    Args:
        alpha: (n_kraus, N_disp) Kraus coefficients
        beta: (n_kraus, N_disp) Kraus displacements
        c_0, d_0: logical |0> coefficients and positions
        c_1, d_1: logical |1> coefficients and positions
        gamma: loss parameter
        n_theta, n_phi: grid resolution

    Returns:
        min_F: worst-case fidelity
        worst_theta, worst_phi: angles of worst-case state
    """
    thetas = jnp.linspace(0, jnp.pi, n_theta)
    phis = jnp.linspace(0, 2 * jnp.pi, n_phi, endpoint=False)

    # Vectorize over grid
    def compute_fidelity_at_angles(theta, phi):
        c_psi, d_psi = bloch_state_coeffs(theta, phi, c_0, d_0, c_1, d_1)
        return state_fidelity_single_coherent(alpha, beta, c_psi, d_psi, gamma)

    # Create meshgrid and flatten
    Theta, Phi = jnp.meshgrid(thetas, phis, indexing='ij')
    Theta_flat = Theta.ravel()
    Phi_flat = Phi.ravel()

    # Compute fidelities for all grid points
    fidelities = jax.vmap(compute_fidelity_at_angles)(Theta_flat, Phi_flat)

    min_idx = jnp.argmin(fidelities)
    min_F = fidelities[min_idx]
    worst_theta = Theta_flat[min_idx]
    worst_phi = Phi_flat[min_idx]

    return min_F, worst_theta, worst_phi


@jax.jit
# ============================================================
# ENTANGLEMENT FIDELITY (AVERAGE-CASE, FOR COMPARISON)
# ============================================================

@jax.jit
def entanglement_fidelity_displacement(alpha, beta, c_0, d_0, c_1, d_1, gamma):
    """
    Compute entanglement fidelity directly in the coherent basis.

    Fe = (1/4) sum_{mu,nu} sum_k <psi_mu| R_k E(|psi_mu><psi_nu|) R_k† |psi_nu>

    This is the average fidelity metric (related to avg by F_avg = (2*Fe + 1)/3).
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
# SOFT-MINIMUM FOR DIFFERENTIABLE WORST-CASE
# ============================================================

@partial(jax.jit, static_argnums=(7, 8))
def logsumexp_min_fidelity(
    alpha, beta, c_0, d_0, c_1, d_1, gamma,
    n_theta=12, n_phi=24, temperature=0.05
):
    """
    Differentiable min via log-sum-exp trick.

    min(F) ~ -T * log(sum_i exp(-F_i/T))

    More numerically stable than softmin.
    """
    thetas = jnp.linspace(0.01, jnp.pi - 0.01, n_theta)
    phis = jnp.linspace(0, 2 * jnp.pi, n_phi, endpoint=False)

    def compute_fidelity_at_angles(theta, phi):
        c_psi, d_psi = bloch_state_coeffs(theta, phi, c_0, d_0, c_1, d_1)
        return state_fidelity_single_coherent(alpha, beta, c_psi, d_psi, gamma)

    Theta, Phi = jnp.meshgrid(thetas, phis, indexing='ij')
    fidelities = jax.vmap(
        lambda t, p: compute_fidelity_at_angles(t, p)
    )(Theta.ravel(), Phi.ravel())

    # Log-sum-exp for minimum
    lse_min = -temperature * jax.scipy.special.logsumexp(-fidelities / temperature)

    return lse_min


# ============================================================
# WORST-CASE CMA-ES OPTIMIZATION
# ============================================================

def optimize_worstcase_cmaes(
    logical_0, logical_1, gamma,
    N_depth=6,
    n_restarts=5,
    popsize=80,
    maxiter=1500,
    sigma0=3.0,
    n_theta=16,
    n_phi=32,
    verbose=True,
):
    """
    CMA-ES optimization for worst-case fidelity.

    Maximizes the minimum fidelity over all Bloch sphere states.

    Args:
        logical_0, logical_1: CoherentKet GKP logical states
        gamma: loss parameter
        N_depth: CD+R circuit depth
        n_restarts: number of CMA-ES restarts
        popsize: CMA-ES population size
        maxiter: max generations per restart
        sigma0: initial step size
        n_theta, n_phi: Bloch sphere grid resolution
        verbose: print progress

    Returns:
        best_params: (N_depth, 4) optimized circuit parameters
        best_Fwc: best worst-case fidelity
        info: dict with optimization details
    """
    import cma

    N_l = 2 ** N_depth
    d_half = float(jnp.sqrt(jnp.pi / 2))

    def unpack(x_real):
        p = jnp.zeros((N_depth, 4), dtype=jnp.complex64)
        for i in range(N_depth):
            p = p.at[i, 0].set(x_real[4*i] + 1j * x_real[4*i+1])
            p = p.at[i, 1].set(x_real[4*i+2])
            p = p.at[i, 2].set(x_real[4*i+3])
        return p

    @jax.jit
    def eval_worstcase(p_complex):
        alpha, beta = g(p_complex, N_l)
        Fwc, _, _ = worst_case_fidelity_grid(
            alpha, beta,
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds,
            gamma, n_theta, n_phi
        )
        return Fwc

    # JIT warmup
    _ = eval_worstcase(jnp.zeros((N_depth, 4), dtype=jnp.complex64))

    def objective(x):
        return -float(eval_worstcase(unpack(np.array(x))))

    # Baseline: identity circuit
    Fwc_id = -objective(np.zeros(N_depth * 4))

    if verbose:
        print(f"Worst-case CMA-ES: N_depth={N_depth}, N_l={N_l}, "
              f"restarts={n_restarts}, pop={popsize}")
        print(f"  Fwc_id={Fwc_id:.6f}")
        sys.stdout.flush()

    best_fwc = Fwc_id
    best_x = np.zeros(N_depth * 4)
    trials = []
    t_total = time.time()

    for trial in range(n_restarts):
        # GKP-informed initial point
        x0 = np.zeros(N_depth * 4)
        x0[0] = d_half
        x0[3] = np.pi / 2
        if N_depth > 1:
            x0[5] = d_half
            x0[7] = np.pi / 2

        # Add small random perturbation
        x0 += 0.3 * np.random.randn(N_depth * 4)

        es = cma.CMAEvolutionStrategy(x0, sigma0, {
            'maxiter': maxiter, 'popsize': popsize,
            'verbose': -1, 'seed': trial, 'tolfun': 1e-9,
        })

        gen = 0
        t0 = time.time()
        while not es.stop():
            solutions = es.ask()
            fitnesses = [objective(x) for x in solutions]
            es.tell(solutions, fitnesses)
            gen += 1

        fwc = -es.result.fbest
        elapsed = time.time() - t0
        improved = fwc > Fwc_id + 0.001

        trials.append({'seed': trial, 'Fwc': fwc, 'gens': gen, 'time': elapsed})

        if verbose:
            flag = ' ***' if improved else ''
            print(f"  trial {trial:2d}: Fwc={fwc:.6f} "
                  f"({gen} gens, {elapsed:.0f}s){flag}")
            sys.stdout.flush()

        if fwc > best_fwc:
            best_fwc = fwc
            best_x = es.result.xbest.copy()

    elapsed_total = time.time() - t_total
    best_params = unpack(best_x)
    n_improved = sum(1 for t in trials if t['Fwc'] > Fwc_id + 0.001)

    if verbose:
        print(f"\n  Best Fwc={best_fwc:.6f} (+{best_fwc-Fwc_id:+.6f})")
        print(f"  Improved: {n_improved}/{n_restarts} trials")
        print(f"  Total time: {elapsed_total:.0f}s")
        sys.stdout.flush()

    return best_params, best_fwc, {
        'Fwc_id': Fwc_id, 'trials': trials,
        'n_improved': n_improved,
        'total_time': elapsed_total,
    }


# ============================================================
# HYBRID CMA-ES + GRADIENT FOR WORST-CASE
# ============================================================

def hybrid_worstcase_optimization(
    logical_0, logical_1, gamma,
    N_depth=6,
    cma_popsize=80,
    cma_maxiter=1000,
    cma_restarts=3,
    grad_steps=2000,
    grad_lr=0.002,
    n_theta=16,
    n_phi=32,
    temperature=0.03,
    sigma0=3.0,
    verbose=True,
):
    """
    Hybrid CMA-ES + gradient descent for worst-case optimization.

    Phase 1: CMA-ES explores landscape to find good basin
    Phase 2: Gradient descent (using soft-min) for fine-tuning

    Args:
        logical_0, logical_1: CoherentKet GKP logical states
        gamma: loss parameter
        N_depth: CD+R circuit depth
        cma_popsize, cma_maxiter, cma_restarts: CMA-ES parameters
        grad_steps, grad_lr: gradient descent parameters
        n_theta, n_phi: Bloch sphere grid resolution
        temperature: softmin temperature
        sigma0: CMA-ES step size
        verbose: print progress

    Returns:
        best_params: (N_depth, 4) optimized parameters
        best_Fwc: best worst-case fidelity
        info: dict with optimization details
    """
    N_l = 2 ** N_depth

    # Phase 1: CMA-ES
    if verbose:
        print(f"\n--- Phase 1: CMA-ES (N_depth={N_depth}) ---")
        sys.stdout.flush()

    params_cma, Fwc_cma, cma_info = optimize_worstcase_cmaes(
        logical_0, logical_1, gamma,
        N_depth=N_depth, n_restarts=cma_restarts,
        popsize=cma_popsize, maxiter=cma_maxiter,
        n_theta=n_theta, n_phi=n_phi,
        sigma0=sigma0, verbose=verbose,
    )
    Fwc_id = cma_info['Fwc_id']

    # Phase 2: Gradient fine-tuning
    if verbose:
        print(f"\n--- Phase 2: Gradient fine-tuning ---")
        print(f"  Starting from CMA-ES Fwc={Fwc_cma:.6f}")
        sys.stdout.flush()

    def loss_fn(params):
        alpha, beta = g(params, N_l)
        soft_min = logsumexp_min_fidelity(
            alpha, beta,
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds,
            gamma, n_theta, n_phi, temperature
        )
        return (1.0 - soft_min).real

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    @jax.jit
    def true_worstcase(params):
        alpha, beta = g(params, N_l)
        Fwc, _, _ = worst_case_fidelity_grid(
            alpha, beta,
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds,
            gamma, n_theta, n_phi
        )
        return Fwc

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=grad_lr * 0.01,
        peak_value=grad_lr,
        warmup_steps=grad_steps // 20,
        decay_steps=grad_steps,
        end_value=grad_lr * 0.001,
    )
    optimizer = optax.adam(schedule)
    params = jnp.array(params_cma)
    opt_state = optimizer.init(params)

    best_Fwc = float(true_worstcase(params))
    best_params = jnp.array(params)
    t_start = time.time()

    for step in range(grad_steps):
        loss, grads = grad_fn(params)
        updates, opt_state = optimizer.update(jnp.conj(grads), opt_state)
        params = optax.apply_updates(params, updates)
        params = params.at[:, 3].set(jnp.zeros(N_depth))

        cur_Fwc = float(true_worstcase(params))
        if cur_Fwc > best_Fwc:
            best_Fwc = cur_Fwc
            best_params = jnp.array(params)

        if verbose and step % 500 == 0:
            elapsed = time.time() - t_start
            print(f"  step {step}: Fwc={cur_Fwc:.6f} "
                  f"(best={best_Fwc:.6f}) [{elapsed:.0f}s]")
            sys.stdout.flush()

    elapsed = time.time() - t_start

    if verbose:
        print(f"  Fine-tuning done ({elapsed:.0f}s): Fwc={best_Fwc:.6f}")
        print(f"  CMA-ES: {Fwc_cma:.6f} -> Fine-tuned: {best_Fwc:.6f} "
              f"(delta={best_Fwc-Fwc_cma:+.6f})")
        print(f"  Improvement over identity: {best_Fwc-Fwc_id:+.6f}")
        sys.stdout.flush()

    return best_params, best_Fwc, {
        **cma_info,
        'Fwc_cma': Fwc_cma,
        'Fwc_final': best_Fwc,
        'grad_steps': grad_steps,
    }


# ============================================================
# COMPARISON: WORST-CASE vs AVERAGE-CASE
# ============================================================

def compare_worstcase_vs_average(
    logical_0, logical_1, gamma,
    N_depth=6,
    params_worstcase=None,
    params_avgcase=None,
    n_theta=24,
    n_phi=48,
    n_rounds=5,
    verbose=True,
):
    """
    Compare worst-case optimized vs average-case optimized recovery.

    Evaluates:
    - Worst-case fidelity
    - Average (entanglement) fidelity
    - Multi-round stability

    Args:
        logical_0, logical_1: CoherentKet GKP logical states
        gamma: loss parameter
        N_depth: circuit depth
        params_worstcase: parameters from worst-case optimization
        params_avgcase: parameters from average-case optimization
        n_theta, n_phi: high-resolution grid for evaluation
        n_rounds: number of error correction rounds to simulate
        verbose: print results

    Returns:
        dict with comparison results
    """
    N_l = 2 ** N_depth

    @jax.jit
    def compute_all_metrics(params):
        alpha, beta = g(params, N_l)

        # Worst-case fidelity
        Fwc, worst_theta, worst_phi = worst_case_fidelity_grid(
            alpha, beta,
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds,
            gamma, n_theta, n_phi
        )

        # Average fidelity
        Fe = entanglement_fidelity_displacement(
            alpha, beta,
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds,
            gamma
        )

        return Fwc, Fe, worst_theta, worst_phi

    results = {}

    if params_worstcase is not None:
        Fwc_wc, Fe_wc, theta_wc, phi_wc = compute_all_metrics(params_worstcase)
        results['worstcase_opt'] = {
            'Fwc': float(Fwc_wc),
            'Fe': float(Fe_wc),
            'worst_theta': float(theta_wc),
            'worst_phi': float(phi_wc),
        }
        if verbose:
            print(f"\nWorst-case optimized:")
            print(f"  Fwc = {Fwc_wc:.6f} (at theta={theta_wc:.3f}, phi={phi_wc:.3f})")
            print(f"  Fe  = {Fe_wc:.6f}")

    if params_avgcase is not None:
        Fwc_avg, Fe_avg, theta_avg, phi_avg = compute_all_metrics(params_avgcase)
        results['avgcase_opt'] = {
            'Fwc': float(Fwc_avg),
            'Fe': float(Fe_avg),
            'worst_theta': float(theta_avg),
            'worst_phi': float(phi_avg),
        }
        if verbose:
            print(f"\nAverage-case optimized:")
            print(f"  Fwc = {Fwc_avg:.6f} (at theta={theta_avg:.3f}, phi={phi_avg:.3f})")
            print(f"  Fe  = {Fe_avg:.6f}")

    # Identity baseline
    params_id = jnp.zeros((N_depth, 4), jnp.complex64)
    Fwc_id, Fe_id, _, _ = compute_all_metrics(params_id)
    results['identity'] = {
        'Fwc': float(Fwc_id),
        'Fe': float(Fe_id),
    }
    if verbose:
        print(f"\nIdentity (no recovery):")
        print(f"  Fwc = {Fwc_id:.6f}")
        print(f"  Fe  = {Fe_id:.6f}")

    # Multi-round analysis: estimate effective error rate
    if verbose and params_worstcase is not None and params_avgcase is not None:
        print(f"\nMulti-round stability estimate ({n_rounds} rounds):")

        # Worst-case optimized: error accumulates as Fwc^n
        Fwc_multi_wc = Fwc_wc ** n_rounds
        Fe_multi_wc = Fe_wc ** n_rounds
        print(f"  Worst-case opt: Fwc^{n_rounds} = {Fwc_multi_wc:.6f}, "
              f"Fe^{n_rounds} = {Fe_multi_wc:.6f}")

        # Average-case optimized
        Fwc_multi_avg = Fwc_avg ** n_rounds
        Fe_multi_avg = Fe_avg ** n_rounds
        print(f"  Avg-case opt:   Fwc^{n_rounds} = {Fwc_multi_avg:.6f}, "
              f"Fe^{n_rounds} = {Fe_multi_avg:.6f}")

        results['multiround'] = {
            'n_rounds': n_rounds,
            'worstcase_Fwc_multi': float(Fwc_multi_wc),
            'worstcase_Fe_multi': float(Fe_multi_wc),
            'avgcase_Fwc_multi': float(Fwc_multi_avg),
            'avgcase_Fe_multi': float(Fe_multi_avg),
        }

    return results


# ============================================================
# FOCK-BASIS CROSS-VALIDATION
# ============================================================

def evaluate_worstcase_fock(params, logical_0, logical_1, gamma,
                            n_theta=16, n_phi=32, loss_rank=10, N=GKP_N):
    """
    Evaluate worst-case fidelity in Fock basis for cross-validation.

    Args:
        params: (N_depth, 4) circuit parameters
        logical_0, logical_1: CoherentKet GKP states
        gamma: loss parameter
        n_theta, n_phi: Bloch sphere grid
        loss_rank: number of loss Kraus operators
        N: Fock space dimension

    Returns:
        Fwc_fock: worst-case fidelity in Fock basis
    """
    N_depth = params.shape[0]
    N_l = 2 ** N_depth

    # Build recovery operators in Fock basis
    alpha, beta = g(params, N_l)
    recovery_ops = channel_from_b(alpha, beta)

    # Build loss operators
    loss_ops = make_pureloss_fock(gamma, rank=loss_rank, N=N)

    # Build Fock logical states
    def build_fock_state(ck):
        coherents = jnp.squeeze(
            jax.vmap(lambda a: dqcoherent(N, a))(ck.ds)
        )
        if coherents.ndim == 3:
            coherents = coherents.squeeze(-1)
        psi = jnp.einsum('bn,b->n', coherents, ck.cs).reshape(-1, 1)
        psi = psi / jnp.sqrt(jnp.real(dag(psi) @ psi).squeeze())
        return psi

    psi_0 = build_fock_state(logical_0)
    psi_1 = build_fock_state(logical_1)

    # Grid search over Bloch sphere
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    min_F = 1.0
    for theta in thetas:
        for phi in phis:
            # Bloch state
            c0 = np.cos(theta / 2)
            c1 = np.sin(theta / 2) * np.exp(1j * phi)
            psi = c0 * psi_0 + c1 * psi_1
            psi = psi / jnp.sqrt(jnp.real(dag(psi) @ psi).squeeze())

            # Density matrix
            rho = psi @ dag(psi)

            # Apply loss then recovery
            rho_after_loss = apply_kraus_map_nonorm(loss_ops, rho)
            rho_after_recovery = apply_kraus_map_nonorm(recovery_ops, rho_after_loss)

            # State fidelity
            F = float(jnp.real(dag(psi) @ rho_after_recovery @ psi).squeeze())
            min_F = min(min_F, F)

    return min_F


# ============================================================
# MAIN SCRIPT
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Worst-Case Fidelity Optimizer")
    print("=" * 60)

    # Parameters
    gamma = 0.1
    Delta = 0.3
    N_depth = 6

    print(f"\nParameters: gamma={gamma}, Delta={Delta}, N_depth={N_depth}")

    # Build GKP states
    logical_0 = gkp_coherent_dm(mu=0, N_trunc=3, Delta=Delta, lattice='square')
    logical_1 = gkp_coherent_dm(mu=1, N_trunc=3, Delta=Delta, lattice='square')

    print(f"Logical states: |0_L| has {logical_0.cs.shape[0]} coherent states")

    # Compute baselines
    N_l = 2 ** N_depth
    id_alpha = jnp.ones((1, 1), dtype=jnp.complex64)
    id_beta = jnp.zeros((1, 1), dtype=jnp.complex64)

    Fe_id = float(entanglement_fidelity_displacement(
        id_alpha, id_beta,
        logical_0.cs, logical_0.ds,
        logical_1.cs, logical_1.ds, gamma
    ))

    Fwc_id, _, _ = worst_case_fidelity_grid(
        id_alpha, id_beta,
        logical_0.cs, logical_0.ds,
        logical_1.cs, logical_1.ds,
        gamma, 16, 32
    )
    Fwc_id = float(Fwc_id)

    print(f"\nIdentity baselines:")
    print(f"  Fe (entanglement fidelity) = {Fe_id:.6f}")
    print(f"  Fwc (worst-case fidelity) = {Fwc_id:.6f}")

    # Run worst-case optimization
    print(f"\n{'='*60}")
    print("Running Hybrid Worst-Case Optimization")
    print(f"{'='*60}")

    params_wc, Fwc_opt, wc_info = hybrid_worstcase_optimization(
        logical_0, logical_1, gamma,
        N_depth=N_depth,
        cma_popsize=60,
        cma_maxiter=800,
        cma_restarts=3,
        grad_steps=1500,
        grad_lr=0.002,
        n_theta=12,
        n_phi=24,
        temperature=0.03,
        verbose=True,
    )

    # Fock cross-validation
    print(f"\n{'='*60}")
    print("Fock-Basis Cross-Validation")
    print(f"{'='*60}")

    Fwc_fock = evaluate_worstcase_fock(
        params_wc, logical_0, logical_1, gamma,
        n_theta=16, n_phi=32
    )
    print(f"  Worst-case Fwc (coherent): {Fwc_opt:.6f}")
    print(f"  Worst-case Fwc (Fock):     {Fwc_fock:.6f}")
    print(f"  Difference:                {abs(Fwc_opt - Fwc_fock):.6f}")

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  gamma = {gamma}")
    print(f"  Identity baseline:       Fwc={Fwc_id:.6f}, Fe={Fe_id:.6f}")
    print(f"  Worst-case optimized:    Fwc={Fwc_opt:.6f}")
    print(f"\n  Worst-case improvement:  +{Fwc_opt - Fwc_id:.6f}")
