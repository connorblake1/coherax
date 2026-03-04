"""
fixedpoint_depth_sweep_claude.py

Depth sweep using Fixed-Point objective (||S - I_4||_F^2) for comparison.

This runs in parallel with the Direct Fidelity depth sweep to compare
the two optimization objectives across circuit depths.

Usage:
    python -m gkp_utils.fixedpoint_depth_sweep_claude
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import sys
import os

from gkp_utils.characteristic_jax_utils import (
    gkp_coherent_dm, g, channel_from_b,
    make_pureloss_fock, dqdag, GKP_N,
    coherent_overlap, aOmegab,
)
from gkp_utils.transpose_channel_claude import (
    build_gkp_states, coherent_ket_to_fock,
    entanglement_fidelity_no_recovery,
)
from gkp_utils.coherent_tree_optimizer_claude import (
    entanglement_fidelity_displacement,
)


# Configuration
GAMMA = 0.05
DELTA = 0.3
N_TRUNC = 3
DEPTHS = [5, 6, 7, 8]
POPSIZE = 100
MAXITER = 1200
N_SEEDS = 3


def compute_superoperator_coherent(alpha, beta, logical_0, logical_1, gamma):
    """
    Compute the 4x4 Liouville superoperator S of R o E on the 2D code space.
    """
    t = jnp.sqrt(1.0 - gamma)
    r = jnp.sqrt(gamma)
    n_kraus = alpha.shape[0]

    psi = [logical_0, logical_1]
    S = jnp.zeros((4, 4), dtype=jnp.complex64)

    for mu in range(2):
        for nu in range(2):
            i = 2 * mu + nu
            for mu_p in range(2):
                for nu_p in range(2):
                    j = 2 * mu_p + nu_p
                    val = _compute_matrix_element(
                        alpha, beta,
                        psi[mu], psi[nu], psi[mu_p], psi[nu_p],
                        t, r
                    )
                    S = S.at[i, j].set(val)

    return S


def _compute_matrix_element(alpha, beta, psi_mu, psi_nu, psi_mu_p, psi_nu_p, t, r):
    """Compute <ψ_μ| R o E (|ψ_μ'><ψ_ν'|) |ψ_ν>"""
    n_kraus = alpha.shape[0]

    c_mu, d_mu = psi_mu.cs, psi_mu.ds
    c_nu, d_nu = psi_nu.cs, psi_nu.ds
    c_mu_p, d_mu_p = psi_mu_p.cs, psi_mu_p.ds
    c_nu_p, d_nu_p = psi_nu_p.cs, psi_nu_p.ds

    val = 0.0 + 0.0j

    for k in range(n_kraus):
        L_mu_p = _compute_L(alpha[k], beta[k], c_mu, d_mu, c_mu_p, d_mu_p, t)
        L_nu_p = _compute_L(alpha[k], beta[k], c_nu, d_nu, c_nu_p, d_nu_p, t)

        env_ovlp = coherent_overlap(
            r * d_nu_p.reshape(-1, 1),
            r * d_mu_p.reshape(1, -1),
        )

        v_mu_p = c_mu_p * L_mu_p
        v_nu_p = c_nu_p * L_nu_p

        val += jnp.conj(v_nu_p) @ env_ovlp @ v_mu_p

    return val


def _compute_L(alpha_k, beta_k, c_out, d_out, c_in, d_in, t):
    """Compute L[a'] = <ψ_out| R_k |t·d_in[a']>"""
    td_in = t * d_in

    phase = jnp.exp(-1j * aOmegab(
        beta_k[:, None],
        td_in[None, :],
    ))

    shifted = beta_k[:, None] + td_in[None, :]

    ovlp = coherent_overlap(
        d_out[:, None, None],
        shifted[None, :, :],
    )

    L = jnp.einsum('p,j,ja,pja->a', jnp.conj(c_out), alpha_k, phase, ovlp)

    return L


def fixedpoint_loss(params, N_l, logical_0, logical_1, gamma):
    """Fixed-Point loss: ||S - I_4||_F^2"""
    alpha, beta = g(params, N_l)
    S = compute_superoperator_coherent(alpha, beta, logical_0, logical_1, gamma)
    I4 = jnp.eye(4, dtype=jnp.complex64)
    return jnp.real(jnp.sum(jnp.abs(S - I4)**2))


def optimize_fixedpoint_cmaes(
    logical_0, logical_1, gamma,
    N_depth, popsize=100, maxiter=1200, sigma0=2.5,
    seed=42, verbose=True,
):
    """CMA-ES optimization with Fixed-Point objective."""
    import cma

    N_l = 2 ** N_depth
    n_params = N_depth * 4
    d_half = float(jnp.sqrt(jnp.pi / 2))

    def unpack(x_real):
        p = jnp.zeros((N_depth, 4), dtype=jnp.complex64)
        for i in range(N_depth):
            p = p.at[i, 0].set(x_real[4*i] + 1j * x_real[4*i+1])
            p = p.at[i, 1].set(x_real[4*i+2])
            p = p.at[i, 2].set(x_real[4*i+3])
        return p

    @jax.jit
    def eval_loss(p_complex):
        return fixedpoint_loss(p_complex, N_l, logical_0, logical_1, gamma)

    @jax.jit
    def eval_fe(p_complex):
        alpha, beta = g(p_complex, N_l)
        return entanglement_fidelity_displacement(
            alpha, beta,
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds, gamma)

    # JIT warmup
    _ = eval_loss(jnp.zeros((N_depth, 4), dtype=jnp.complex64))

    def objective(x):
        return float(eval_loss(unpack(np.array(x))))

    # Initial point
    x0 = np.zeros(n_params)
    x0[0] = d_half
    x0[3] = np.pi/2
    if N_depth > 1:
        x0[5] = d_half
        x0[7] = np.pi/2

    loss_id = float(eval_loss(jnp.zeros((N_depth, 4), dtype=jnp.complex64)))
    Fe_id = float(eval_fe(jnp.zeros((N_depth, 4), dtype=jnp.complex64)))

    if verbose:
        print(f"  N_depth={N_depth}, N_l={N_l}, params={n_params}")
        print(f"  Loss_id={loss_id:.6f}, Fe_id={Fe_id:.6f}")
        sys.stdout.flush()

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'maxiter': maxiter, 'popsize': popsize,
        'verbose': -1, 'seed': seed, 'tolfun': 1e-9,
    })

    gen = 0
    t_start = time.time()

    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)

        if verbose and gen % 300 == 0:
            best_loss = es.result.fbest
            best_params = unpack(es.result.xbest)
            best_fe = float(eval_fe(best_params))
            elapsed = time.time() - t_start
            print(f"    gen {gen}: loss={best_loss:.6f}, Fe={best_fe:.6f} [{elapsed:.0f}s]")
            sys.stdout.flush()
        gen += 1

    best_loss = es.result.fbest
    best_params = unpack(es.result.xbest)
    best_fe = float(eval_fe(best_params))
    elapsed = time.time() - t_start

    if verbose:
        print(f"  Done ({elapsed:.0f}s): loss={best_loss:.6f}, Fe={best_fe:.6f}")
        print(f"  Fe improvement: {best_fe - Fe_id:+.6f}")
        sys.stdout.flush()

    return best_params, best_fe, {
        'loss': best_loss,
        'Fe_id': Fe_id,
        'loss_id': loss_id,
        'elapsed': elapsed,
    }


def run_fixedpoint_depth_sweep(gamma=GAMMA, depths=DEPTHS, n_seeds=N_SEEDS, verbose=True):
    """Run Fixed-Point optimization across depths."""

    logical_0, logical_1 = build_gkp_states(Delta=DELTA, N_trunc=N_TRUNC)
    psi_0 = coherent_ket_to_fock(logical_0, GKP_N)
    psi_1 = coherent_ket_to_fock(logical_1, GKP_N)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(dqdag(psi_0) @ psi_0).squeeze())
    psi_1 = psi_1 / jnp.sqrt(jnp.real(dqdag(psi_1) @ psi_1).squeeze())

    loss_ops = make_pureloss_fock(gamma, rank=10, N=GKP_N)
    Fe_none = float(entanglement_fidelity_no_recovery(loss_ops, psi_0, psi_1))

    if verbose:
        print("=" * 70)
        print(f"Fixed-Point Depth Sweep: gamma={gamma}, depths={depths}")
        print(f"Fe(none)={Fe_none:.6f}")
        print("=" * 70)
        sys.stdout.flush()

    results = {
        'gamma': gamma,
        'Fe_none': Fe_none,
        'depths': [],
        'Fe_best': [],
        'loss_best': [],
        'params_best': {},
    }

    for depth in depths:
        if verbose:
            print(f"\n{'='*50}")
            print(f"N_depth = {depth}")
            print(f"{'='*50}")
            sys.stdout.flush()

        best_fe = 0
        best_loss = float('inf')
        best_params = None

        for seed in range(n_seeds):
            if verbose:
                print(f"\n--- Seed {seed} ---")
                sys.stdout.flush()

            params, fe, info = optimize_fixedpoint_cmaes(
                logical_0, logical_1, gamma,
                N_depth=depth, popsize=POPSIZE, maxiter=MAXITER,
                sigma0=2.5, seed=seed, verbose=verbose,
            )

            if fe > best_fe:
                best_fe = fe
                best_loss = info['loss']
                best_params = params

        results['depths'].append(depth)
        results['Fe_best'].append(best_fe)
        results['loss_best'].append(best_loss)
        results['params_best'][depth] = np.array(best_params)

        if verbose:
            print(f"\n  Summary depth {depth}:")
            print(f"    Best Fe: {best_fe:.6f}, Best Loss: {best_loss:.6f}")
            print(f"    Improvement over none: +{best_fe - Fe_none:.4f}")
            sys.stdout.flush()

    return results


def print_summary(results):
    """Print summary table."""
    print("\n" + "=" * 70)
    print("FIXED-POINT DEPTH SWEEP SUMMARY")
    print("=" * 70)
    print(f"gamma = {results['gamma']}, Fe(none) = {results['Fe_none']:.6f}")
    print("-" * 70)
    print(f"{'Depth':>6} | {'N_l':>6} | {'Best Fe':>10} | {'Δ':>8} | {'Loss':>10}")
    print("-" * 70)

    for i, depth in enumerate(results['depths']):
        N_l = 2 ** depth
        fe = results['Fe_best'][i]
        loss = results['loss_best'][i]
        delta = fe - results['Fe_none']
        marker = " *" if fe == max(results['Fe_best']) else ""
        print(f"{depth:6d} | {N_l:6d} | {fe:10.6f} | {delta:+8.4f} | {loss:10.6f}{marker}")

    print("=" * 70)


def save_results(results, filepath='results/fixedpoint_depth_sweep.npz'):
    """Save results to NPZ."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    save_dict = {
        'gamma': results['gamma'],
        'Fe_none': results['Fe_none'],
        'depths': np.array(results['depths']),
        'Fe_best': np.array(results['Fe_best']),
        'loss_best': np.array(results['loss_best']),
    }
    for depth, params in results['params_best'].items():
        save_dict[f'params_depth_{depth}'] = params

    np.savez(filepath, **save_dict)
    print(f"\nSaved to {filepath}")


if __name__ == "__main__":
    results = run_fixedpoint_depth_sweep(
        gamma=0.05,
        depths=[5, 6, 7, 8],
        n_seeds=2,
        verbose=True
    )
    print_summary(results)
    save_results(results)
