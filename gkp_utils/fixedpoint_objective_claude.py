"""
fixedpoint_objective_claude.py

Fixed-Point optimization objective: minimize ||S - I_4||_F^2
where S is the Liouville superoperator of R o E on the code space.

This provides a comparison to Direct Fidelity optimization.
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from gkp_utils.characteristic_jax_utils import (
    gkp_coherent_dm, g, coherent_overlap, aOmegab,
)


def compute_superoperator_coherent(alpha, beta, logical_0, logical_1, gamma):
    """
    Compute the 4x4 Liouville superoperator S of R o E on the 2D code space.
    
    The superoperator acts on vectorized density matrices:
        vec(rho') = S @ vec(rho)
    where rho is in the {|0_L>, |1_L|}⊗{<0_L|, <1_L|} basis.
    
    Args:
        alpha: (n_kraus, N_disp) recovery Kraus coefficients
        beta: (n_kraus, N_disp) recovery displacement positions
        logical_0, logical_1: CoherentKet objects
        gamma: loss parameter
        
    Returns:
        S: (4, 4) superoperator matrix
    """
    t = jnp.sqrt(1.0 - gamma)
    r = jnp.sqrt(gamma)
    n_kraus = alpha.shape[0]
    
    psi = [logical_0, logical_1]
    
    # S[i,j] = sum_k <ψ_μ| R_k E(|ψ_μ'><ψ_ν'|) R_k† |ψ_ν>
    # where i = 2*μ + ν, j = 2*μ' + ν'
    S = jnp.zeros((4, 4), dtype=jnp.complex64)
    
    for mu in range(2):
        for nu in range(2):
            i = 2 * mu + nu
            for mu_p in range(2):
                for nu_p in range(2):
                    j = 2 * mu_p + nu_p
                    
                    # Compute <ψ_μ| R o E (|ψ_μ'><ψ_ν'|) |ψ_ν>
                    val = _compute_matrix_element(
                        alpha, beta,
                        psi[mu], psi[nu], psi[mu_p], psi[nu_p],
                        t, r
                    )
                    S = S.at[i, j].set(val)
    
    return S


def _compute_matrix_element(alpha, beta, psi_mu, psi_nu, psi_mu_p, psi_nu_p, t, r):
    """
    Compute <ψ_μ| R o E (|ψ_μ'><ψ_ν'|) |ψ_ν>
    
    Using coherent basis expansion and loss channel action.
    """
    n_kraus = alpha.shape[0]
    
    # Get coherent state data
    c_mu, d_mu = psi_mu.cs, psi_mu.ds
    c_nu, d_nu = psi_nu.cs, psi_nu.ds
    c_mu_p, d_mu_p = psi_mu_p.cs, psi_mu_p.ds
    c_nu_p, d_nu_p = psi_nu_p.cs, psi_nu_p.ds
    
    val = 0.0 + 0.0j
    
    for k in range(n_kraus):
        # Compute L_k^{(μ')} and L_k^{(ν')}
        # L_k^{(μ')}[a'] = <ψ_μ| R_k |t·d_μ'[a']>
        # where |t·d_μ'[a']> is the attenuated coherent state
        
        L_mu_p = _compute_L(alpha[k], beta[k], c_mu, d_mu, c_mu_p, d_mu_p, t)
        L_nu_p = _compute_L(alpha[k], beta[k], c_nu, d_nu, c_nu_p, d_nu_p, t)
        
        # Environment overlap: <r·d_ν'[b'] | r·d_μ'[a']>
        env_ovlp = coherent_overlap(
            r * d_nu_p.reshape(-1, 1),
            r * d_mu_p.reshape(1, -1),
        )
        
        # Sum over input coherent states
        v_mu_p = c_mu_p * L_mu_p
        v_nu_p = c_nu_p * L_nu_p
        
        val += jnp.conj(v_nu_p) @ env_ovlp @ v_mu_p
    
    return val


def _compute_L(alpha_k, beta_k, c_out, d_out, c_in, d_in, t):
    """
    Compute L[a'] = <ψ_out| R_k |t·d_in[a']>
    
    where R_k = sum_j alpha_k[j] D(beta_k[j])
    """
    A_out = d_out.shape[0]
    A_in = d_in.shape[0]
    N_disp = alpha_k.shape[0]
    
    td_in = t * d_in  # (A_in,)
    
    # phase[j, a'] = exp(-i * aOmegab(beta_k[j], t*d_in[a']))
    phase = jnp.exp(-1j * aOmegab(
        beta_k[:, None],
        td_in[None, :],
    ))  # (N_disp, A_in)
    
    # shifted[j, a'] = beta_k[j] + t*d_in[a']
    shifted = beta_k[:, None] + td_in[None, :]  # (N_disp, A_in)
    
    # ovlp[p, j, a'] = <d_out[p] | shifted[j, a']>
    ovlp = coherent_overlap(
        d_out[:, None, None],
        shifted[None, :, :],
    )  # (A_out, N_disp, A_in)
    
    # L[a'] = sum_{p,j} conj(c_out[p]) * alpha_k[j] * phase[j,a'] * ovlp[p,j,a']
    L = jnp.einsum('p,j,ja,pja->a', jnp.conj(c_out), alpha_k, phase, ovlp)
    
    return L


def fixedpoint_loss(params, N_l, logical_0, logical_1, gamma):
    """
    Fixed-Point loss: ||S - I_4||_F^2
    
    Args:
        params: (N_depth, 4) circuit parameters
        N_l: number of displacement terms (2^N_depth)
        logical_0, logical_1: CoherentKet objects
        gamma: loss parameter
        
    Returns:
        loss: scalar Frobenius distance squared
    """
    alpha, beta = g(params, N_l)
    S = compute_superoperator_coherent(alpha, beta, logical_0, logical_1, gamma)
    I4 = jnp.eye(4, dtype=jnp.complex64)
    return jnp.real(jnp.sum(jnp.abs(S - I4)**2))


def optimize_fixedpoint_cmaes(
    logical_0, logical_1, gamma,
    N_depth=6, popsize=80, maxiter=2000, sigma0=3.0,
    seed=42, verbose=True,
):
    """
    CMA-ES optimization with Fixed-Point objective.
    """
    import cma
    import time
    import sys
    
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
    
    # Also compute Fe for comparison
    @jax.jit
    def eval_fe(p_complex):
        from gkp_utils.coherent_tree_optimizer_claude import entanglement_fidelity_displacement
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
        print(f"Fixed-Point CMA-ES: N_depth={N_depth}, N_l={N_l}")
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
        
        if verbose and gen % 200 == 0:
            best_loss = es.result.fbest
            best_params = unpack(es.result.xbest)
            best_fe = float(eval_fe(best_params))
            elapsed = time.time() - t_start
            print(f"  gen {gen}: loss={best_loss:.6f}, Fe={best_fe:.6f} [{elapsed:.0f}s]")
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


if __name__ == "__main__":
    from gkp_utils.transpose_channel_claude import build_gkp_states
    
    print("Testing Fixed-Point objective...")
    
    logical_0, logical_1 = build_gkp_states(Delta=0.3, N_trunc=3)
    
    for gamma in [0.05, 0.10]:
        print(f"\n=== gamma = {gamma} ===")
        params, Fe, info = optimize_fixedpoint_cmaes(
            logical_0, logical_1, gamma,
            N_depth=6, popsize=80, maxiter=1000,
            verbose=True,
        )
