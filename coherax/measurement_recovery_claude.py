"""
measurement_recovery_claude.py

Measurement-conditioned recovery for GKP stabilization, like real SBS.

Instead of tracing out the ancilla unconditionally (sum over outcomes),
this module implements measurement feedback where:
- We measure the ancilla qubit
- Based on outcome 0 or 1, apply different correction displacements
- The effective Kraus operators become: K_i' = D(d_i) @ K_i

For a tree of depth T measurements:
- 2^T possible measurement outcomes (bitstrings)
- 2^T correction displacements to optimize
- Effective channel: rho -> sum_m K_m' @ rho @ K_m'†

This is what real SBS does - it measures the ancilla and applies corrections
based on the outcome to keep the state in the GKP codespace.

Optimization is via CMA-ES since correction displacements add discrete
structure that makes gradient descent less effective.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import jax.random as jr
import numpy as np
from functools import partial
from jaxtyping import Array
from typing import Tuple, Dict, Any
import sys

from coherax.characteristic_jax_utils import (
    CoherentKet,
    gkp_coherent_dm,
    super_g, g,
    apply_kraus_map_nonorm, apply_kraus_map,
    compose_channel_kraus, channel_from_b,
    make_pureloss_fock, make_transpose_for_pureloss,
    GKP_N, dqdag, dqtrace, dqcoherent, dqdisplace, dqeye,
    dqtensor, sigma_x, sigma_z, dqdestroy, dqcreate,
    aOmegab, e_n1iaOmegab,
    root2,
)


# ============================================================
# GKP STATE SETUP
# ============================================================

def build_gkp_states(Delta=0.3, N_trunc=3, lattice="square"):
    """Build GKP logical states as CoherentKet objects."""
    logical_0 = gkp_coherent_dm(mu=0, N_trunc=N_trunc, Delta=Delta, lattice=lattice)
    logical_1 = gkp_coherent_dm(mu=1, N_trunc=N_trunc, Delta=Delta, lattice=lattice)
    return logical_0, logical_1


def coherent_ket_to_fock(ck, N=GKP_N):
    """Convert CoherentKet to Fock-basis ket."""
    coherents = jax.vmap(lambda alpha: dqcoherent(N, alpha))(ck.ds)
    return jnp.einsum('ijk,i->jk', coherents, ck.cs)


# ============================================================
# SBS UNITARY CONSTRUCTION (matching transpose_channel_claude.py)
# ============================================================

def _cd_royer(beta, N=GKP_N):
    """Controlled Displacement in Royer convention."""
    a_hat = dqdestroy(N)
    a_dag_hat = dqcreate(N)
    generator = dqtensor(
        beta * a_dag_hat - jnp.conj(beta) * a_hat,
        sigma_z
    ) / (2 * root2)
    return jla.expm(generator)


def _r_x(theta):
    """Qubit rotation R_x(theta) = exp(-i*theta*sigma_x/2)."""
    return jla.expm(-1j * theta * sigma_x / 2)


def build_sbs_unitary(direction, Delta=0.3, N=GKP_N):
    """
    Build SBS unitary for one stabilizer direction.

    U_SBS = CD_A_small @ (I x Rx^dag) @ CD_B @ (I x Rx) @ CD_A_small
    """
    GKP_L = 2.0 * jnp.sqrt(jnp.pi)
    alpha_arr = GKP_L * jnp.array([0.0, 1.0], dtype=jnp.complex64)
    beta_arr = GKP_L * jnp.array([-1.0, 0.0], dtype=jnp.complex64)

    c_D = jnp.cosh(Delta**2)
    s_D = jnp.sinh(Delta**2)

    l_j = jnp.sqrt(jnp.abs(alpha_arr)**2 + jnp.abs(beta_arr)**2)
    theta_j = jnp.angle(alpha_arr + 1j * beta_arr)
    epsilon_j = s_D * 4 * jnp.pi / l_j

    j = direction
    eps_j = epsilon_j[j] * jnp.exp(1j * theta_j[j])
    big_disp = -1j * (alpha_arr[j] + 1j * beta_arr[j]) * c_D

    I_N = dqeye(N)
    cd_a_small = _cd_royer(eps_j / 2.0, N)
    cd_b = _cd_royer(big_disp, N)
    rx = _r_x(jnp.pi / 2.0)
    rx_dag = dqdag(rx)

    U = cd_a_small @ (dqtensor(I_N, rx_dag) @ (cd_b @ (dqtensor(I_N, rx) @ cd_a_small)))
    return U


# ============================================================
# MEASUREMENT-CONDITIONED KRAUS OPERATORS
# ============================================================

@partial(jax.jit, static_argnums=1)
def traceout_unitary_separate(U: Array, N=GKP_N) -> Tuple[Array, Array]:
    """
    Extract separate Kraus operators for each measurement outcome.

    K_0 = <0_anc| U |0_anc>  (outcome 0)
    K_1 = <1_anc| U |0_anc>  (outcome 1)

    Returns:
        (K_0, K_1): Each (N, N) array
    """
    import dynamiqs as dq
    K_0 = (dqtensor(jnp.eye(N), dqdag(dq.fock(2, 0)))
           @ U
           @ dqtensor(jnp.eye(N), dq.fock(2, 0)))
    K_1 = (dqtensor(jnp.eye(N), dqdag(dq.fock(2, 1)))
           @ U
           @ dqtensor(jnp.eye(N), dq.fock(2, 0)))
    return K_0, K_1


def build_measurement_kraus_sbs(direction, Delta=0.3, N=GKP_N):
    """
    Build measurement-conditioned Kraus operators for one SBS round.

    Returns:
        K_0: (N, N) Kraus operator for outcome 0
        K_1: (N, N) Kraus operator for outcome 1
    """
    U = build_sbs_unitary(direction, Delta=Delta, N=N)
    return traceout_unitary_separate(U, N)


def apply_correction_displacement(K, d, N=GKP_N):
    """Apply correction displacement D(d) to Kraus operator: K' = D(d) @ K."""
    D = jnp.squeeze(dqdisplace(N, d))
    return D @ K


def build_corrected_kraus_sbs(direction, d_0, d_1, Delta=0.3, N=GKP_N):
    """
    Build SBS Kraus operators with measurement-dependent corrections.

    K_0' = D(d_0) @ K_0  (for outcome 0)
    K_1' = D(d_1) @ K_1  (for outcome 1)

    Returns:
        (2, N, N) array of corrected Kraus operators
    """
    K_0, K_1 = build_measurement_kraus_sbs(direction, Delta=Delta, N=N)
    K_0_corr = apply_correction_displacement(K_0, d_0, N)
    K_1_corr = apply_correction_displacement(K_1, d_1, N)
    return jnp.stack([K_0_corr, K_1_corr])


def build_tree_corrected_kraus(directions, corrections, Delta=0.3, N=GKP_N):
    """
    Build corrected Kraus operators for a tree of measurements.

    For T measurements (len(directions) = T):
    - 2^T total outcomes (all bitstrings of length T)
    - corrections: (2^T,) complex array of correction displacements

    The final Kraus operator for outcome bitstring b = b_{T-1}...b_1 b_0 is:
        K_b = D(corrections[b]) @ K_{b_{T-1}}^{(T-1)} @ ... @ K_{b_0}^{(0)}

    Returns:
        (2^T, N, N) array of Kraus operators
    """
    T = len(directions)
    num_outcomes = 2 ** T

    # Build base Kraus operators for each round
    K_all = []
    for t, d in enumerate(directions):
        K_0, K_1 = build_measurement_kraus_sbs(d, Delta=Delta, N=N)
        K_all.append((K_0, K_1))

    # Build composed operators for each outcome
    final_kraus = jnp.zeros((num_outcomes, N, N), dtype=jnp.complex64)

    for outcome in range(num_outcomes):
        # outcome encodes measurement results as a bitstring
        K = jnp.eye(N, dtype=jnp.complex64)
        for t in range(T):
            bit = (outcome >> t) & 1
            K = K_all[t][bit] @ K
        # Apply correction displacement
        D = jnp.squeeze(dqdisplace(N, corrections[outcome]))
        final_kraus = final_kraus.at[outcome].set(D @ K)

    return final_kraus


# ============================================================
# CD+R CIRCUIT WITH CORRECTIONS (coherent basis)
# ============================================================

def apply_displacement_to_coherent_kraus(alpha, beta, d):
    """
    Apply correction displacement D(d) to a coherent-basis Kraus operator.

    The coherent-basis Kraus operator is: K = sum_i alpha_i D(beta_i)
    After correction: K' = D(d) @ K = sum_i alpha_i D(d) D(beta_i)
                                     = sum_i alpha_i e^{-i Im(d* beta_i)} D(d + beta_i)

    Returns:
        alpha_new, beta_new: Corrected coefficients and displacements
    """
    phase = jnp.exp(-1j * aOmegab(d, beta))
    return alpha * phase, beta + d


def build_corrected_coherent_kraus(circuit_params, corrections, N_l, T_depth):
    """
    Build corrected Kraus operators in the coherent basis.

    Args:
        circuit_params: (T_depth, N_depth, 4) circuit parameters
        corrections: (2^T_depth,) complex correction displacements
        N_l: 2^N_depth
        T_depth: number of traceout rounds

    Returns:
        alpha_corr: (2^T_depth, N_l^T_depth) corrected coefficients
        beta_corr: (2^T_depth, N_l^T_depth) corrected displacements
    """
    # Get uncorrected Kraus operators
    alpha, beta = super_g(circuit_params, N_l=N_l, T=T_depth)

    # Apply corrections to each outcome
    num_outcomes = 2 ** T_depth
    alpha_corr = jnp.zeros_like(alpha)
    beta_corr = jnp.zeros_like(beta)

    for i in range(num_outcomes):
        a_new, b_new = apply_displacement_to_coherent_kraus(
            alpha[i], beta[i], corrections[i]
        )
        alpha_corr = alpha_corr.at[i].set(a_new)
        beta_corr = beta_corr.at[i].set(b_new)

    return alpha_corr, beta_corr


# ============================================================
# FIDELITY COMPUTATION
# ============================================================

def entanglement_fidelity_fock(recovery_ops, loss_ops, psi_0, psi_1):
    """
    Compute entanglement fidelity in Fock basis.

    F_e = (1/4) sum_{mu,nu} <mu_L| R(E(|mu_L><nu_L|)) |nu_L>
    """
    psi = [psi_0, psi_1]
    F_e = 0.0
    for mu in range(2):
        for nu in range(2):
            rho_mn = psi[mu] @ dqdag(psi[nu])
            after_loss = apply_kraus_map_nonorm(loss_ops, rho_mn)
            after_recovery = apply_kraus_map_nonorm(recovery_ops, after_loss)
            F_e += (dqdag(psi[mu]) @ after_recovery @ psi[nu]).squeeze()
    return jnp.real(F_e) / 4.0


@jax.jit
def analytic_corrected_fidelity_single(
    alpha: Array, beta: Array,
    c: Array, d: Array,
    gamma: float,
):
    """
    Analytic average fidelity for a single input state (given by c, d).

    For corrected Kraus operators K_i = sum_j alpha_{ij} D(beta_{ij}),
    computes the fidelity of the recovery channel after pure loss.

    This is essentially analytic_pureloss_recovery_fidelity_thetaphi from
    characteristic_jax_utils.py, but we call it here for clarity.
    """
    from coherax.characteristic_jax_utils import (
        analytic_pureloss_recovery_fidelity_thetaphi
    )
    return analytic_pureloss_recovery_fidelity_thetaphi(
        alpha=alpha, beta=beta, c=c, d=d, gamma=gamma
    )


def make_corrected_loss_fn(logical_0, logical_1, gamma, N_l, T_depth, n_points=64, seed=42):
    """
    Build a loss function for optimizing circuit params + corrections.

    Parameters to optimize:
        circuit_params: (T_depth, N_depth, 4) -- circuit parameters
        corrections: (2^T_depth,) complex -- correction displacements

    We pack these into a single array for optimization:
        params_flat = [circuit_params.ravel(), corrections.ravel()]
    """
    key = jr.PRNGKey(seed)
    u = jr.uniform(key, (2, n_points))
    thetas = jnp.arccos(2 * u[0] - 1.0)
    phis = 2.0 * jnp.pi * u[1]
    c0s = jnp.cos(thetas / 2)
    c1s = jnp.sin(thetas / 2) * jnp.exp(1.0j * phis)

    N_depth = int(np.log2(N_l))
    num_corrections = 2 ** T_depth
    circuit_size = T_depth * N_depth * 4

    @jax.jit
    def loss_fn(params_flat):
        # Unpack parameters
        circuit_flat = params_flat[:circuit_size]
        corrections_flat = params_flat[circuit_size:]

        circuit_params = circuit_flat.reshape((T_depth, N_depth, 4))
        corrections = corrections_flat[:num_corrections]

        # Build corrected Kraus operators
        alpha, beta = build_corrected_coherent_kraus(
            circuit_params, corrections, N_l, T_depth
        )

        def fid_for_point(c0, c1):
            cs = jnp.concatenate([c0 * logical_0.cs, c1 * logical_1.cs])
            ds = jnp.concatenate([logical_0.ds, logical_1.ds])
            return analytic_corrected_fidelity_single(
                alpha=alpha, beta=beta, c=cs, d=ds, gamma=gamma
            )

        return 1.0 - jnp.mean(jax.vmap(fid_for_point)(c0s, c1s))

    return loss_fn, circuit_size, num_corrections


# ============================================================
# CMA-ES OPTIMIZER
# ============================================================

def optimize_with_cmaes(
    logical_0, logical_1, gamma,
    T_depth=1, N_depth=6,
    popsize=20, maxiter=500,
    sigma0=2.0, verbose=True,
):
    """
    Optimize circuit params + corrections using CMA-ES.

    CMA-ES is better than gradient descent here because:
    1. The correction displacements introduce discrete structure
    2. The loss landscape has many local minima
    3. CMA-ES handles complex-valued parameters naturally (as 2D real)

    Returns:
        best_circuit_params, best_corrections, best_loss
    """
    try:
        import cma
    except ImportError:
        print("CMA-ES requires 'cma' package. Install with: pip install cma")
        return None, None, 1.0

    N_l = 2 ** N_depth
    num_corrections = 2 ** T_depth
    circuit_size = T_depth * N_depth * 4

    # Build loss function
    loss_fn, _, _ = make_corrected_loss_fn(
        logical_0, logical_1, gamma, N_l, T_depth,
        n_points=64, seed=42
    )

    # Initial point (random)
    key = jr.PRNGKey(np.random.randint(10000))
    k1, k2, k3, k4 = jr.split(key, 4)

    # Circuit params initialization
    circuit_init = jnp.zeros((T_depth, N_depth, 4), jnp.complex64)
    circuit_init = circuit_init.at[:, :, 0].set(
        2.0 * jr.normal(k1, (T_depth, N_depth)) +
        2.0j * jr.normal(k2, (T_depth, N_depth))
    )
    circuit_init = circuit_init.at[:, :, 1:3].set(
        jnp.pi * jr.uniform(k3, (T_depth, N_depth, 2))
    )

    # Corrections initialization (small random)
    corrections_init = 0.5 * (jr.normal(k4, (num_corrections,)) +
                              1j * jr.normal(jr.split(k4)[0], (num_corrections,)))

    # Pack into real vector for CMA-ES
    # Complex -> (real, imag) pairs
    def pack_params(circuit_params, corrections):
        circuit_flat = circuit_params.ravel()
        corrections_flat = corrections.ravel()
        params = jnp.concatenate([circuit_flat, corrections_flat])
        return jnp.concatenate([jnp.real(params), jnp.imag(params)])

    def unpack_params(x):
        x = jnp.array(x)
        n = len(x) // 2
        params_complex = x[:n] + 1j * x[n:]
        circuit_flat = params_complex[:circuit_size]
        corrections = params_complex[circuit_size:circuit_size + num_corrections]
        return circuit_flat, corrections

    def objective(x):
        circuit_flat, corrections = unpack_params(x)
        params_flat = jnp.concatenate([circuit_flat, corrections])
        return float(loss_fn(params_flat))

    x0 = np.array(pack_params(circuit_init, corrections_init))

    # Run CMA-ES
    opts = {
        'popsize': popsize,
        'maxiter': maxiter,
        'verb_disp': 100 if verbose else 0,
        'verb_log': 0,
        'tolfun': 1e-6,
    }

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_x = x0
    best_loss = objective(x0)

    while not es.stop():
        solutions = es.ask()
        fitness = [objective(x) for x in solutions]
        es.tell(solutions, fitness)

        if min(fitness) < best_loss:
            best_loss = min(fitness)
            best_x = solutions[fitness.index(best_loss)]
            if verbose:
                print(f"  New best: 1-F = {best_loss:.6f}")
                sys.stdout.flush()

    # Unpack best solution
    circuit_flat, corrections = unpack_params(best_x)
    circuit_params = circuit_flat.reshape((T_depth, N_depth, 4))

    return circuit_params, corrections, best_loss


# ============================================================
# GRADIENT-BASED OPTIMIZER (alternative)
# ============================================================

def optimize_with_adam(
    logical_0, logical_1, gamma,
    T_depth=1, N_depth=6,
    lr=0.005, steps=3000, restarts=3,
    random_dist=2.0, verbose=True,
):
    """
    Optimize circuit params + corrections using Adam.

    Gradient-based alternative to CMA-ES.
    """
    import optax
    import equinox as eqx

    N_l = 2 ** N_depth
    num_corrections = 2 ** T_depth
    circuit_size = T_depth * N_depth * 4

    loss_fn, _, _ = make_corrected_loss_fn(
        logical_0, logical_1, gamma, N_l, T_depth,
        n_points=64, seed=42
    )

    grad_fn = jax.value_and_grad(lambda p: loss_fn(p).real)

    best_loss = 1.0
    best_circuit = None
    best_corrections = None

    for restart in range(restarts):
        key = jr.PRNGKey(np.random.randint(10000))
        k1, k2, k3, k4 = jr.split(key, 4)

        # Initialize
        circuit_init = jnp.zeros((T_depth, N_depth, 4), jnp.complex64)
        circuit_init = circuit_init.at[:, :, 0].set(
            random_dist * jr.normal(k1, (T_depth, N_depth)) +
            random_dist * 1j * jr.normal(k2, (T_depth, N_depth))
        )
        circuit_init = circuit_init.at[:, :, 1:3].set(
            jnp.pi * jr.uniform(k3, (T_depth, N_depth, 2))
        )

        corrections_init = 0.5 * (jr.normal(k4, (num_corrections,)) +
                                  1j * jr.normal(jr.split(k4)[0], (num_corrections,)))

        params = jnp.concatenate([circuit_init.ravel(), corrections_init])

        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)

        for step in range(steps):
            params_c = params.astype(jnp.complex64)
            loss, grads = grad_fn(params_c)
            updates, opt_state = optimizer.update(jnp.conj(grads), opt_state)
            params = optax.apply_updates(params, updates)

            # Zero out gamma parameter
            circuit_flat = params[:circuit_size].reshape((T_depth, N_depth, 4))
            circuit_flat = circuit_flat.at[:, :, 3].set(0.0)
            params = params.at[:circuit_size].set(circuit_flat.ravel())

            if step % 500 == 0 and verbose:
                print(f"  Restart {restart}, Step {step}: 1-F = {loss:.6f}")
                sys.stdout.flush()

        final_loss = float(loss_fn(params))
        if verbose:
            print(f"  Restart {restart} final: 1-F = {final_loss:.6f}")

        if final_loss < best_loss:
            best_loss = final_loss
            circuit_params = params[:circuit_size].reshape((T_depth, N_depth, 4))
            best_circuit = circuit_params
            best_corrections = params[circuit_size:circuit_size + num_corrections]
            if verbose:
                print(f"  >> New best! 1-F = {best_loss:.6f}")

    return best_circuit, best_corrections, best_loss


# ============================================================
# COMPARISON: UNCONDITIONAL VS MEASUREMENT-CONDITIONED
# ============================================================

def compare_recoveries(
    gamma=0.05, Delta=0.3, N_trunc=3,
    T_depth=1, N_depth=6,
    loss_rank=10, N=GKP_N, verbose=True,
):
    """
    Compare unconditional traceout vs measurement-conditioned recovery.

    1. No recovery baseline
    2. Unconditional traceout (current approach)
    3. Measurement-conditioned with optimized corrections
    4. Transpose channel (theoretical optimum)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Comparing recovery strategies: gamma={gamma}, Delta={Delta}")
        print(f"Circuit: T_depth={T_depth}, N_depth={N_depth}")
        print(f"{'='*60}")

    # Build GKP states
    logical_0, logical_1 = build_gkp_states(Delta=Delta, N_trunc=N_trunc)
    psi_0 = coherent_ket_to_fock(logical_0, N)
    psi_1 = coherent_ket_to_fock(logical_1, N)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(dqdag(psi_0) @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(dqdag(psi_1) @ psi_1).squeeze())

    # Loss channel
    loss_ops = make_pureloss_fock(gamma, rank=loss_rank, N=N)

    # 1. No recovery
    Fe_none = entanglement_fidelity_fock(
        jnp.eye(N)[None, :, :], loss_ops, psi_0, psi_1
    )
    if verbose:
        print(f"  No recovery:      F_e = {Fe_none:.6f}")

    # 4. Transpose channel (compute first for reference)
    transpose_ops = make_transpose_for_pureloss(loss_ops, logical_0, logical_1)
    Fe_transpose = entanglement_fidelity_fock(transpose_ops, loss_ops, psi_0, psi_1)
    if verbose:
        print(f"  Transpose bound:  F_e = {Fe_transpose:.6f}")

    # 2. Unconditional traceout (optimize without corrections)
    if verbose:
        print("\nOptimizing unconditional traceout...")

    from coherax.recovery_optimizer_claude import optimize_recovery
    uncond_params, uncond_loss, _ = optimize_recovery(
        logical_0, logical_1, gamma,
        T_depth=T_depth, N_depth=N_depth,
        lr=0.003, steps=2000, restarts=2,
        batch_size=32, verbose=False,
    )

    N_l = 2 ** N_depth
    alpha_uncond, beta_uncond = super_g(uncond_params, N_l=N_l, T=T_depth)
    uncond_ops = channel_from_b(alpha_uncond, beta_uncond)
    Fe_uncond = entanglement_fidelity_fock(uncond_ops, loss_ops, psi_0, psi_1)
    if verbose:
        print(f"  Unconditional:    F_e = {Fe_uncond:.6f}")

    # 3. Measurement-conditioned with optimized corrections
    if verbose:
        print("\nOptimizing measurement-conditioned with corrections...")

    cond_circuit, cond_corrections, cond_loss = optimize_with_adam(
        logical_0, logical_1, gamma,
        T_depth=T_depth, N_depth=N_depth,
        lr=0.005, steps=2000, restarts=2,
        verbose=False,
    )

    alpha_cond, beta_cond = build_corrected_coherent_kraus(
        cond_circuit, cond_corrections, N_l, T_depth
    )
    cond_ops = channel_from_b(alpha_cond, beta_cond)
    Fe_cond = entanglement_fidelity_fock(cond_ops, loss_ops, psi_0, psi_1)
    if verbose:
        print(f"  Meas-conditioned: F_e = {Fe_cond:.6f}")

    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print("SUMMARY:")
        print(f"  No recovery:        F_e = {Fe_none:.6f}")
        print(f"  Unconditional:      F_e = {Fe_uncond:.6f} (improvement: {Fe_uncond - Fe_none:.6f})")
        print(f"  Meas-conditioned:   F_e = {Fe_cond:.6f} (improvement: {Fe_cond - Fe_none:.6f})")
        print(f"  Transpose bound:    F_e = {Fe_transpose:.6f}")
        print(f"  Gap (cond - uncond): {Fe_cond - Fe_uncond:.6f}")
        print(f"  Gap to bound:        {Fe_transpose - Fe_cond:.6f}")
        print(f"{'='*60}")

    return {
        'gamma': gamma,
        'Delta': Delta,
        'T_depth': T_depth,
        'N_depth': N_depth,
        'Fe_none': float(Fe_none),
        'Fe_uncond': float(Fe_uncond),
        'Fe_cond': float(Fe_cond),
        'Fe_transpose': float(Fe_transpose),
        'uncond_params': uncond_params,
        'cond_circuit': cond_circuit,
        'cond_corrections': cond_corrections,
    }


# ============================================================
# MULTI-ROUND FIDELITY (cascaded recovery)
# ============================================================

def multi_round_fidelity(recovery_ops, loss_ops, psi_0, psi_1, n_rounds=5):
    """
    Compute fidelity after n rounds of (loss, recovery).

    Returns:
        fidelities: (n_rounds,) array of F_e after each round
    """
    fidelities = []

    # Start with maximally entangled state
    rho_00 = psi_0 @ dqdag(psi_0)
    rho_01 = psi_0 @ dqdag(psi_1)
    rho_10 = psi_1 @ dqdag(psi_0)
    rho_11 = psi_1 @ dqdag(psi_1)
    rhos = [[rho_00, rho_01], [rho_10, rho_11]]

    for r in range(n_rounds):
        # Apply loss then recovery
        for mu in range(2):
            for nu in range(2):
                after_loss = apply_kraus_map_nonorm(loss_ops, rhos[mu][nu])
                rhos[mu][nu] = apply_kraus_map_nonorm(recovery_ops, after_loss)

        # Compute F_e
        F_e = 0.0
        for mu in range(2):
            for nu in range(2):
                F_e += (dqdag(psi_0 if mu == 0 else psi_1) @
                        rhos[mu][nu] @
                        (psi_0 if nu == 0 else psi_1)).squeeze()
        fidelities.append(float(jnp.real(F_e) / 4.0))

    return jnp.array(fidelities)


def compare_multi_round(
    gamma=0.05, Delta=0.3, N_trunc=3,
    T_depth=1, N_depth=6, n_rounds=5,
    verbose=True,
):
    """
    Compare multi-round performance of different recovery strategies.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Multi-round comparison: gamma={gamma}, {n_rounds} rounds")
        print(f"{'='*60}")

    # Build states and ops
    logical_0, logical_1 = build_gkp_states(Delta=Delta, N_trunc=N_trunc)
    psi_0 = coherent_ket_to_fock(logical_0)
    psi_1 = coherent_ket_to_fock(logical_1)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(dqdag(psi_0) @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(dqdag(psi_1) @ psi_1).squeeze())

    loss_ops = make_pureloss_fock(gamma, rank=10)

    # Optimize both strategies
    N_l = 2 ** N_depth

    from coherax.recovery_optimizer_claude import optimize_recovery
    uncond_params, _, _ = optimize_recovery(
        logical_0, logical_1, gamma,
        T_depth=T_depth, N_depth=N_depth,
        steps=1500, restarts=2, verbose=False,
    )
    alpha_uncond, beta_uncond = super_g(uncond_params, N_l=N_l, T=T_depth)
    uncond_ops = channel_from_b(alpha_uncond, beta_uncond)

    cond_circuit, cond_corrections, _ = optimize_with_adam(
        logical_0, logical_1, gamma,
        T_depth=T_depth, N_depth=N_depth,
        steps=1500, restarts=2, verbose=False,
    )
    alpha_cond, beta_cond = build_corrected_coherent_kraus(
        cond_circuit, cond_corrections, N_l, T_depth
    )
    cond_ops = channel_from_b(alpha_cond, beta_cond)

    # Multi-round fidelities
    Fe_uncond_multi = multi_round_fidelity(uncond_ops, loss_ops, psi_0, psi_1, n_rounds)
    Fe_cond_multi = multi_round_fidelity(cond_ops, loss_ops, psi_0, psi_1, n_rounds)
    Fe_none_multi = multi_round_fidelity(
        jnp.eye(GKP_N)[None, :, :], loss_ops, psi_0, psi_1, n_rounds
    )

    if verbose:
        print("\nFidelity over rounds:")
        print(f"{'Round':>6} | {'No recov':>10} | {'Unconditional':>13} | {'Meas-cond':>10}")
        print("-" * 50)
        for r in range(n_rounds):
            print(f"{r+1:6d} | {Fe_none_multi[r]:10.6f} | {Fe_uncond_multi[r]:13.6f} | {Fe_cond_multi[r]:10.6f}")

    return {
        'rounds': list(range(1, n_rounds + 1)),
        'Fe_none': Fe_none_multi,
        'Fe_uncond': Fe_uncond_multi,
        'Fe_cond': Fe_cond_multi,
    }


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("MEASUREMENT-CONDITIONED RECOVERY FOR GKP STABILIZATION")
    print("="*70)
    print()
    print("This implements the key idea from real SBS:")
    print("  - Measure the ancilla qubit")
    print("  - Apply outcome-dependent correction displacements")
    print("  - Effective Kraus ops: K_i' = D(d_i) @ K_i")
    print()

    # Single-round comparison
    results_single = compare_recoveries(
        gamma=0.05, Delta=0.3, N_trunc=3,
        T_depth=1, N_depth=6,
        verbose=True,
    )

    # Multi-round comparison
    print()
    results_multi = compare_multi_round(
        gamma=0.05, Delta=0.3, N_trunc=3,
        T_depth=1, N_depth=6, n_rounds=5,
        verbose=True,
    )

    print()
    print("="*70)
    print("CONCLUSION:")
    print("="*70)
    print("Measurement-conditioned recovery (with optimized correction")
    print("displacements) can outperform unconditional traceout because:")
    print("  1. It uses measurement information adaptively")
    print("  2. Different outcomes indicate different error syndromes")
    print("  3. Targeted corrections can better restore the codespace")
    print()
    print("This is the fundamental principle behind SBS stabilization.")
