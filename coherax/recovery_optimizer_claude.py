"""
recovery_optimizer_claude.py

Optimizes CD+R circuit parameters to approximate the transpose channel
recovery for finite GKP codes under pure loss.

Uses the analytic average fidelity infrastructure from characteristic_jax_utils.py
to optimize circuit parameters via gradient descent.

The circuit ansatz is T_depth traceout rounds, each with N_depth CD+R layers,
giving 2^T_depth Kraus operators with 2^N_depth displacement terms each.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import equinox as eqx
from functools import partial
from jaxtyping import Array
from typing import Any

from coherax.characteristic_jax_utils import (
    CoherentKet,
    gkp_coherent_dm,
    super_g,
    analytic_pureloss_recovery_fidelity_thetaphi,
    analytic_pureloss_recovery_fidelity_random_wrapper,
    make_pureloss_fock, make_transpose_for_pureloss,
    apply_kraus_map_nonorm, channel_from_b,
    GKP_N, dqdag, dqtrace, dqcoherent,
)


# ============================================================
# GKP STATE SETUP
# ============================================================

def build_gkp_states(Delta=0.3, N_trunc=3, lattice="square"):
    logical_0 = gkp_coherent_dm(mu=0, N_trunc=N_trunc, Delta=Delta, lattice=lattice)
    logical_1 = gkp_coherent_dm(mu=1, N_trunc=N_trunc, Delta=Delta, lattice=lattice)
    return logical_0, logical_1


def coherent_ket_to_fock(ck, N=GKP_N):
    coherents = jax.vmap(lambda alpha: dqcoherent(N, alpha))(ck.ds)
    return jnp.einsum('ijk,i->jk', coherents, ck.cs)


# ============================================================
# LOSS FUNCTION: ANALYTIC AVERAGE FIDELITY
# ============================================================

def make_loss_fn(logical_0, logical_1, gamma, N_l, T_depth, batch_size=64):
    """
    Build a differentiable loss function: 1 - F_avg (stochastic).

    Uses stochastic sampling of (theta, phi) on the Bloch sphere to estimate
    F_avg = E[F(theta, phi)].

    Returns:
        loss_fn(circuit_params, key) -> scalar loss (1 - F_avg)
    """

    @jax.jit
    def loss_fn(circuit_params, key):
        alpha, beta = super_g(circuit_params, N_l=N_l, T=T_depth)
        keys = jr.split(key, batch_size)

        def single_fidelity(k):
            u = jr.uniform(k, (2,))
            theta = jnp.arccos(2 * u[0] - 1.0)
            phi = 2.0 * jnp.pi * u[1]
            c0 = jnp.cos(theta / 2)
            c1 = jnp.sin(theta / 2) * jnp.exp(1.0j * phi)
            cs = jnp.concatenate([c0 * logical_0.cs, c1 * logical_1.cs])
            ds = jnp.concatenate([logical_0.ds, logical_1.ds])
            return analytic_pureloss_recovery_fidelity_thetaphi(
                alpha=alpha, beta=beta, c=cs, d=ds, gamma=gamma
            )

        fids = jax.vmap(single_fidelity)(keys)
        return 1.0 - jnp.mean(fids)

    return loss_fn


def make_deterministic_loss_fn(logical_0, logical_1, gamma, N_l, T_depth,
                                n_points=64, seed=42):
    """
    Build a deterministic loss function using fixed Bloch sphere samples.
    Better for reproducibility, evaluation, and faster JIT compilation.
    """
    key = jr.PRNGKey(seed)
    u = jr.uniform(key, (2, n_points))
    thetas = jnp.arccos(2 * u[0] - 1.0)
    phis = 2.0 * jnp.pi * u[1]
    c0s = jnp.cos(thetas / 2)
    c1s = jnp.sin(thetas / 2) * jnp.exp(1.0j * phis)

    @jax.jit
    def loss_fn(circuit_params):
        alpha, beta = super_g(circuit_params, N_l=N_l, T=T_depth)

        def fid_for_point(c0, c1):
            cs = jnp.concatenate([c0 * logical_0.cs, c1 * logical_1.cs])
            ds = jnp.concatenate([logical_0.ds, logical_1.ds])
            return analytic_pureloss_recovery_fidelity_thetaphi(
                alpha=alpha, beta=beta, c=cs, d=ds, gamma=gamma
            )

        return 1.0 - jnp.mean(jax.vmap(fid_for_point)(c0s, c1s))

    return loss_fn


# ============================================================
# OPTIMIZER
# ============================================================

def optimize_recovery(
    logical_0,
    logical_1,
    gamma,
    T_depth=2,
    N_depth=6,
    lr=0.003,
    steps=5000,
    restarts=5,
    batch_size=64,
    random_dist=4.0,
    random_angle=1.0,
    verbose=True,
):
    """
    Optimize CD+R circuit parameters for loss recovery.

    Minimizes 1 - F_avg where F_avg is the average fidelity over Bloch sphere.

    Uses deterministic training (fixed Bloch sphere samples) with a JIT-compiled
    train step for efficient gradient computation.

    Args:
        logical_0, logical_1: CoherentKet GKP logical states
        gamma: loss parameter
        T_depth: number of traceout rounds (2^T_depth Kraus operators)
        N_depth: layers per round (2^N_depth displacements per Kraus operator)
        lr: learning rate
        steps: gradient steps per restart
        restarts: number of random restarts
        batch_size: Bloch sphere samples for training
        random_dist: scale of random initialization for displacements
        random_angle: scale of random initialization for angles
        verbose: print progress

    Returns:
        best_params: (T_depth, N_depth, 4) optimized circuit parameters
        best_loss: final loss value (1 - F_avg)
        loss_history: loss values over training
    """
    import sys
    N_l = 2 ** N_depth

    # Deterministic loss for training (JIT-compiled)
    train_fn = make_deterministic_loss_fn(
        logical_0, logical_1, gamma, N_l, T_depth,
        n_points=batch_size, seed=123
    )
    eval_fn = make_deterministic_loss_fn(
        logical_0, logical_1, gamma, N_l, T_depth,
        n_points=128, seed=42
    )

    # Gradient function (train_fn is JIT, value_and_grad traces through it)
    grad_fn = jax.value_and_grad(lambda p: train_fn(p).real)

    best_loss = 1.0
    best_params = None
    all_loss_history = []

    for restart in range(restarts):
        key = jr.PRNGKey(np.random.randint(10000))
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=lr * 0.1,
            peak_value=lr,
            warmup_steps=steps // 20,
            decay_steps=steps,
            end_value=lr * 0.01,
        )
        optimizer = optax.adam(schedule)

        k1, k2, k3, key = jr.split(key, 4)
        params = jnp.zeros((T_depth, N_depth, 4), jnp.complex64)
        params = params.at[:, :, 1:].set(
            2 * random_angle * jnp.pi * jr.uniform(key=k2, shape=(T_depth, N_depth, 3))
        )
        params = params.at[:, :, 0].set(
            random_dist * jr.normal(key=k1, shape=(T_depth, N_depth))
            + random_dist * 1j * jr.normal(key=k3, shape=(T_depth, N_depth))
        )

        opt_state = optimizer.init(params)
        restart_losses = []
        last_loss = 1.0

        for step in range(steps):
            params_c = params.astype(jnp.complex64)
            loss, grads = grad_fn(params_c)
            updates, opt_state = optimizer.update(jnp.conj(grads), opt_state)
            params = optax.apply_updates(params, updates)
            # Zero out gamma parameter (rotation convention)
            params = params.at[:, :, 3].set(jnp.zeros((T_depth, N_depth)))

            if step % 100 == 0:
                det_loss = eval_fn(params)
                restart_losses.append(float(det_loss))
                if verbose and step % 500 == 0:
                    print(
                        f"  Restart {restart}, Step {step}: "
                        f"1-F_avg = {det_loss:.6f}"
                    )
                    sys.stdout.flush()
                if last_loss == det_loss and det_loss > 0.05:
                    if verbose:
                        print(f"  Restart {restart}: early stop at step {step}")
                        sys.stdout.flush()
                    break
                last_loss = det_loss

        final_loss = float(eval_fn(params))
        all_loss_history.append(restart_losses)
        if verbose:
            print(f"  Restart {restart} final: 1-F_avg = {final_loss:.6f}")
            sys.stdout.flush()

        if final_loss < best_loss:
            best_loss = final_loss
            best_params = params
            if verbose:
                print(f"  >> New best! 1-F_avg = {best_loss:.6f}")
                sys.stdout.flush()

    return best_params, best_loss, all_loss_history


# ============================================================
# FOCK-BASIS VALIDATION
# ============================================================

def validate_in_fock(circuit_params, logical_0, logical_1, gamma,
                     N_l, T_depth, loss_rank=10, N=GKP_N):
    """
    Cross-validate analytic F_avg against Fock-basis entanglement fidelity.

    Returns:
        Fe_analytic: F_e estimated from analytic F_avg via Horodecki
        Fe_fock: F_e from Fock basis (definitive)
    """
    # Analytic F_avg
    eval_fn = make_deterministic_loss_fn(
        logical_0, logical_1, gamma, N_l, T_depth, n_points=256, seed=0
    )
    F_avg = 1.0 - eval_fn(circuit_params)
    Fe_analytic = (3 * F_avg - 1) / 2

    # Fock-basis
    alpha, beta = super_g(circuit_params, N_l=N_l, T=T_depth)
    recovery_ops = channel_from_b(alpha, beta)
    loss_ops = make_pureloss_fock(gamma, rank=loss_rank, N=N)

    psi_0 = coherent_ket_to_fock(logical_0, N)
    psi_1 = coherent_ket_to_fock(logical_1, N)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(dqdag(psi_0) @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(dqdag(psi_1) @ psi_1).squeeze())

    psi = [psi_0, psi_1]
    Fe_fock = 0.0
    for mu in range(2):
        for nu in range(2):
            rho_mn = psi[mu] @ dqdag(psi[nu])
            after_loss = apply_kraus_map_nonorm(loss_ops, rho_mn)
            after_recovery = apply_kraus_map_nonorm(recovery_ops, after_loss)
            Fe_fock += (dqdag(psi[mu]) @ after_recovery @ psi[nu]).squeeze()
    Fe_fock = jnp.real(Fe_fock) / 4.0

    return float(Fe_analytic), float(Fe_fock)


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def optimize_for_gamma(gamma, Delta=0.3, N_trunc=3, T_depth=2, N_depth=6,
                       lr=0.003, steps=5000, restarts=5, batch_size=64):
    """
    Full optimization pipeline for a single gamma value.

    Returns:
        dict with optimization results
    """
    print(f"\n{'='*60}")
    print(f"Optimizing recovery for gamma={gamma}, Delta={Delta}")
    print(f"Circuit: T_depth={T_depth}, N_depth={N_depth} "
          f"({2**T_depth} Kraus ops, {2**N_depth} disps each)")
    print(f"{'='*60}")

    logical_0, logical_1 = build_gkp_states(Delta=Delta, N_trunc=N_trunc)

    best_params, best_loss, loss_history = optimize_recovery(
        logical_0, logical_1, gamma,
        T_depth=T_depth, N_depth=N_depth,
        lr=lr, steps=steps, restarts=restarts, batch_size=batch_size,
    )

    F_avg = 1.0 - best_loss
    print(f"\nBest result: F_avg={F_avg:.6f}")

    # Fock-basis cross-validation
    N_l = 2 ** N_depth
    Fe_analytic, Fe_fock = validate_in_fock(
        best_params, logical_0, logical_1, gamma, N_l, T_depth
    )
    print(f"Cross-validation: Fe_analytic={Fe_analytic:.6f}, Fe_fock={Fe_fock:.6f}")

    # Transpose channel bound
    loss_ops = make_pureloss_fock(gamma, rank=10)
    transpose_ops = make_transpose_for_pureloss(loss_ops, logical_0, logical_1)
    psi_0 = coherent_ket_to_fock(logical_0)
    psi_1 = coherent_ket_to_fock(logical_1)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(dqdag(psi_0) @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(dqdag(psi_1) @ psi_1).squeeze())
    psi = [psi_0, psi_1]
    Fe_transpose = 0.0
    for mu in range(2):
        for nu in range(2):
            rho_mn = psi[mu] @ dqdag(psi[nu])
            after_loss = apply_kraus_map_nonorm(loss_ops, rho_mn)
            after_recovery = apply_kraus_map_nonorm(transpose_ops, after_loss)
            Fe_transpose += (dqdag(psi[mu]) @ after_recovery @ psi[nu]).squeeze()
    Fe_transpose = float(jnp.real(Fe_transpose) / 4.0)

    print(f"Transpose channel bound: Fe_transpose={Fe_transpose:.6f}")
    print(f"Gap to bound: {Fe_transpose - Fe_fock:.6f}")

    return {
        'gamma': gamma,
        'Delta': Delta,
        'T_depth': T_depth,
        'N_depth': N_depth,
        'best_params': best_params,
        'best_loss': best_loss,
        'F_avg': F_avg,
        'Fe_analytic': Fe_analytic,
        'Fe_fock': Fe_fock,
        'Fe_transpose': Fe_transpose,
        'loss_history': loss_history,
    }


if __name__ == "__main__":
    result = optimize_for_gamma(
        gamma=0.05, Delta=0.3, N_trunc=3,
        T_depth=2, N_depth=6,
        lr=0.003, steps=3000, restarts=3, batch_size=32,
    )
    print(f"\nFinal: Fe_optimized={result['Fe_fock']:.6f}, "
          f"Fe_transpose={result['Fe_transpose']:.6f}")
