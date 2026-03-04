"""
Analytic fidelity for preparing Fock-basis states with coherent-basis CD+R circuits.

The key identity is ⟨n|β⟩ = exp(-|β|²/2) β^n / √(n!), which converts the
(α, β) output of g() into a closed-form, JAX-differentiable fidelity for any
target |ψ⟩ = Σ c_n |n⟩.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp
import optax
from functools import partial
from jaxtyping import Array

from coherax import (
    g,
    channel_from_b,
    GKP_N,
)


@partial(jax.jit, static_argnums=2)
def fock_state_inner_product(alphas: Array, betas: Array, m: int) -> float:
    """
    Compute F_m = Σ_j |Σ_i α_{j,i} ⟨m|β_{j,i}⟩|² for pure Fock state |m⟩.

    This is Eq. (8) from the derivation.

    Args:
        alphas: (2, N_l) complex amplitudes from g()
        betas: (2, N_l) complex displacement positions from g()
        m: target Fock state number

    Returns:
        Fidelity F_m ∈ [0, 1]
    """
    # ⟨m|β⟩ = exp(-|β|²/2) · β^m / √(m!)
    # Direct computation avoids log(0) issues at β=0.
    # For m=0: β^0 = 1, so overlap = exp(-|β|²/2).
    # For m>0: β^m → 0 as β→0, which is correct (vacuum has zero overlap with |m>0⟩).
    envelope = jnp.exp(-0.5 * jnp.abs(betas) ** 2)  # (2, N_l)
    monomial = betas ** m  # (2, N_l), complex power
    norm = 1.0 / jnp.sqrt(jnp.exp(jsp.gammaln(m + 1.0)))
    overlaps = envelope * monomial * norm  # (2, N_l)

    # Σ_i α_{j,i} ⟨m|β_{j,i}⟩ for each j
    inner = jnp.sum(alphas * overlaps, axis=1)  # (2,)
    return jnp.sum(jnp.abs(inner) ** 2).real


@jax.jit
def general_fock_inner_product(alphas: Array, betas: Array, coeffs: Array) -> float:
    """
    Compute F = Σ_j |Σ_i α_{j,i} ⟨ψ|β_{j,i}⟩|² for |ψ⟩ = Σ_n c_n |n⟩.

    This is Eq. (7) from the derivation.

    Args:
        alphas: (2, N_l) complex amplitudes from g()
        betas: (2, N_l) complex displacement positions from g()
        coeffs: (N_max+1,) complex Fock coefficients c_n

    Returns:
        Fidelity F ∈ [0, 1]
    """
    N_max = coeffs.shape[0]
    ns = jnp.arange(N_max, dtype=jnp.float32)

    # h_n(β) = exp(-|β|²/2) · β^n / √(n!)
    envelope = jnp.exp(-0.5 * jnp.abs(betas) ** 2)  # (2, N_l)
    # β^n for each n: (N_max, 2, N_l)
    powers = betas[None, :, :] ** ns[:, None, None]
    norms = 1.0 / jnp.sqrt(jnp.exp(jsp.gammaln(ns + 1.0)))  # (N_max,)
    h = envelope[None, :, :] * powers * norms[:, None, None]  # (N_max, 2, N_l)

    # ⟨ψ|β_{j,i}⟩ = Σ_n c_n* h_n(β_{j,i})
    psi_beta = jnp.sum(jnp.conj(coeffs)[:, None, None] * h, axis=0)  # (2, N_l)

    # Σ_i α_{j,i} ⟨ψ|β_{j,i}⟩ for each j
    inner = jnp.sum(alphas * psi_beta, axis=1)  # (2,)
    return jnp.sum(jnp.abs(inner) ** 2).real


@partial(jax.jit, static_argnums=(1, 2))
def analytic_fidelity_fock_state(circuit_params: Array, N_l: int, m: int) -> float:
    """Fidelity of circuit output with Fock state |m⟩."""
    alphas, betas = g(circuit_params, N_l)
    return fock_state_inner_product(alphas, betas, m)


@partial(jax.jit, static_argnums=1)
def analytic_fidelity_fock_general(circuit_params: Array, N_l: int, coeffs: Array) -> float:
    """Fidelity of circuit output with |ψ⟩ = Σ c_n |n⟩."""
    alphas, betas = g(circuit_params, N_l)
    return general_fock_inner_product(alphas, betas, coeffs)


def optimize_fock_state(
    target_n: int,
    N_depth: int = 6,
    lr: float = 0.005,
    steps: int = 10000,
    restarts: int = 50,
    random_dist: float = 0.1,
    random_angle: float = 0.1,
    lr2: float = None,
    steps2: int = 0,
    verbose: bool = True,
):
    """
    Optimize a CD+R circuit to prepare Fock state |target_n⟩ from vacuum.

    Uses two-phase Adam: phase 1 at `lr` for `steps`, then phase 2 at `lr2`
    for `steps2` (warm-started from phase 1 best).

    Args:
        target_n: target Fock state number
        N_depth: circuit depth (N_l = 2^N_depth displacements)
        lr: learning rate for phase 1
        steps: number of Adam steps per restart (phase 1)
        restarts: number of random restarts
        random_dist: std of initial displacement amplitudes
        random_angle: std of initial rotation angles
        lr2: learning rate for phase 2 (default: lr/10)
        steps2: number of Adam steps for phase 2 refinement
        verbose: print progress

    Returns:
        (best_params, best_fidelity, all_fidelities)
    """
    N_l = 2 ** N_depth
    if lr2 is None:
        lr2 = lr / 10.0

    @partial(jax.jit, static_argnums=(0, 1))
    def loss_fn(N_l_s: int, m_s: int, params: Array) -> float:
        return 1.0 - analytic_fidelity_fock_state(params, N_l_s, m_s)

    @partial(jax.jit, static_argnums=(2, 3))
    def train_step(params: Array, opt_state, N_l_s: int, m_s: int):
        params = params.astype(jnp.complex64)
        grads = jax.grad(partial(loss_fn, N_l_s, m_s))(params)
        grads = jnp.conj(grads)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), new_opt_state

    best_loss = 1.0
    best_params = None
    all_fidelities = []

    for restart in range(restarts):
        key = jr.PRNGKey(restart)
        optimizer = optax.adam(lr)
        k1, k2, k3 = jr.split(key, 3)

        # Small random initialization (proven to work better than large)
        a = jnp.zeros((N_depth, 4), jnp.complex64)
        a = a.at[:, 0].set(
            random_dist * jr.normal(k1, (N_depth,))
            + random_dist * 1j * jr.normal(k3, (N_depth,))
        )
        a = a.at[:, 1:3].set(random_angle * jr.normal(k2, (N_depth, 2)))

        opt_state = optimizer.init(a)
        last_loss = 1.0

        for step in range(steps):
            a, opt_state = train_step(a, opt_state, N_l, target_n)
            a = a.at[:, 3].set(0.0)  # kill gammas (single traceout)

            if step % 500 == 0 and verbose:
                current_loss = loss_fn(N_l, target_n, a)
                if step % 2000 == 0:
                    print(f"  Restart {restart:3d}, Step {step:5d}, 1-F = {current_loss:.6e}")
                if jnp.abs(last_loss - current_loss) < 1e-10 and current_loss > 0.01:
                    break
                last_loss = current_loss

        final_loss = loss_fn(N_l, target_n, a)
        all_fidelities.append(1.0 - float(final_loss))

        if final_loss < best_loss:
            best_loss = final_loss
            best_params = a
            if verbose:
                print(f"  ** Restart {restart}: new best 1-F = {best_loss:.6e}")

    # Phase 2 refinement
    if steps2 > 0 and best_params is not None:
        if verbose:
            print(f"\nPhase 2: refining best with lr={lr2} for {steps2} steps")
        optimizer = optax.adam(lr2)
        a = best_params
        opt_state = optimizer.init(a)
        for step in range(steps2):
            a, opt_state = train_step(a, opt_state, N_l, target_n)
            a = a.at[:, 3].set(0.0)
            if step % 1000 == 0 and verbose:
                current_loss = loss_fn(N_l, target_n, a)
                print(f"  Phase 2 step {step:5d}, 1-F = {current_loss:.6e}")
        final_loss = loss_fn(N_l, target_n, a)
        if final_loss < best_loss:
            best_loss = final_loss
            best_params = a

    return best_params, 1.0 - float(best_loss), np.array(all_fidelities)


def cross_validate_fock(circuit_params: Array, target_n: int, N_fock: int = GKP_N):
    """
    Cross-validate Fock state preparation fidelity in Fock basis.

    Builds the full Kraus operators in Fock space and computes
    F = ⟨target_n| ρ_out |target_n⟩ exactly.
    """
    N_l = 2 ** circuit_params.shape[0]
    alphas, betas = g(circuit_params, N_l)
    kraus_ops = channel_from_b(alphas, betas)

    # Apply to vacuum
    vac = jnp.zeros(N_fock, dtype=jnp.complex64)
    vac = vac.at[0].set(1.0)

    rho_out = jnp.zeros((N_fock, N_fock), dtype=jnp.complex64)
    for j in range(kraus_ops.shape[0]):
        psi_j = kraus_ops[j] @ vac
        rho_out = rho_out + jnp.outer(psi_j, jnp.conj(psi_j))

    # Fidelity with |target_n⟩
    return jnp.real(rho_out[target_n, target_n])


def cross_validate_general(circuit_params: Array, coeffs: Array, N_fock: int = GKP_N):
    """
    Cross-validate general Fock superposition fidelity in Fock basis.

    F = ⟨ψ| ρ_out |ψ⟩ where |ψ⟩ = Σ c_n |n⟩.
    """
    N_l = 2 ** circuit_params.shape[0]
    alphas, betas = g(circuit_params, N_l)
    kraus_ops = channel_from_b(alphas, betas)

    vac = jnp.zeros(N_fock, dtype=jnp.complex64)
    vac = vac.at[0].set(1.0)

    rho_out = jnp.zeros((N_fock, N_fock), dtype=jnp.complex64)
    for j in range(kraus_ops.shape[0]):
        psi_j = kraus_ops[j] @ vac
        rho_out = rho_out + jnp.outer(psi_j, jnp.conj(psi_j))

    # Build target ket in Fock basis
    psi_target = jnp.zeros(N_fock, dtype=jnp.complex64)
    psi_target = psi_target.at[: coeffs.shape[0]].set(coeffs)

    return jnp.real(jnp.conj(psi_target) @ rho_out @ psi_target)


def circuit_time_us(circuit_params: Array, chi_hz: float = 2 * np.pi * 50e3, gamma0: float = 20.0):
    """Estimate circuit execution time in microseconds."""
    beta_total = jnp.sum(jnp.abs(circuit_params[:, 0]))
    t_displacement = float(beta_total) / (chi_hz * gamma0)
    t_ancilla = 24e-9 * circuit_params.shape[0]  # 24 ns/layer overhead
    t_gate_min = 48e-9 * circuit_params.shape[0]  # 48 ns minimum gate time
    return max(t_displacement + t_ancilla, t_gate_min) * 1e6


if __name__ == "__main__":
    # Quick test: optimize |2⟩ with 4 layers, 5 restarts
    print("Test: optimizing |2⟩ with N_depth=4, 5 restarts")
    params, fid, all_fids = optimize_fock_state(
        target_n=2, N_depth=4, lr=0.005, steps=5000, restarts=5,
        random_dist=0.1, random_angle=0.1, verbose=True
    )
    print(f"\nBest fidelity: {fid:.8f} (1-F = {1-fid:.2e})")

    # Cross-validate
    fid_fock = cross_validate_fock(params, target_n=2)
    print(f"Fock cross-validation: {fid_fock:.8f} (1-F = {1-float(fid_fock):.2e})")
