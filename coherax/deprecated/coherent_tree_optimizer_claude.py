"""
coherent_tree_optimizer_claude.py

Binary tree decomposition of the coherent-basis transpose channel and
CD+R circuit optimization to implement each tree node measurement.

All optimization runs entirely in the coherent basis using g() — no Fock projection.

Key insight: the tree B operators only act on the support of E(P_C) within V',
which is typically ~14-dimensional (much smaller than A_Vp ~75). We restrict all
computation to this support subspace for tractable optimization.

Pipeline:
  1. Take Kraus operators Y_k from coherent_transpose_claude.py
  2. Project POVM to E(P_C) support, build binary tree in reduced space
  3. Optimize CD+R params in reduced space (coherent basis only)
  4. Assemble full recovery and evaluate in Fock basis
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from functools import partial
from jaxtyping import Array

from coherax.characteristic_jax_utils import (
    CoherentKet,
    BosonicSubspace,
    coherent_overlap,
    aOmegab,
    e_n1iaOmegab,
    dag,
    g,
    GKP_N,
    dqdag,
    dqcoherent,
    make_pureloss_fock,
    apply_kraus_map_nonorm,
    analytic_pureloss_recovery_fidelity_thetaphi,
    channel_from_b,
)
from coherax.binary_tree_utils import BinaryKrausTree, sqrt_psd


# ============================================================
# SUPPORT PROJECTION
# ============================================================

def compute_support_basis(transpose_result, eps=1e-6):
    """
    Compute the orthonormal basis for the support of E(P_C) within V'.

    The tree only needs to act on this support (typically ~14-dimensional),
    not the full V' space (~75-dimensional).

    Args:
        transpose_result: dict from build_coherent_transpose_channel
        eps: eigenvalue cutoff

    Returns:
        V_supp: (A_Vp, n_support) support basis vectors in V' ortho basis
        n_support: dimension of support
        P_red: (B, n_support) reduced projection matrix (T_Vp @ V_supp)
    """
    p_mat = transpose_result['p_mat']
    ds_all = transpose_result['ds_all']
    gamma = transpose_result['gamma']
    subspace_Vp = transpose_result['subspace_Vp']

    r = jnp.sqrt(gamma)
    rd = r * ds_all
    env_ov = coherent_overlap(rd.reshape(-1, 1), rd.reshape(1, -1))
    p_prime = p_mat * jnp.conj(env_ov)

    Tp_Vp = subspace_Vp.Tp
    EPC_ortho = Tp_Vp @ p_prime @ dag(Tp_Vp)
    EPC_ortho = (EPC_ortho + dag(EPC_ortho)) / 2.0

    eigvals, eigvecs = jnp.linalg.eigh(EPC_ortho)
    mask = eigvals > eps
    n_support = int(jnp.sum(mask))

    V_supp = eigvecs[:, mask]  # (A_Vp, n_support)
    P_red = subspace_Vp.T @ V_supp  # (B, n_support)

    return V_supp, n_support, P_red


# ============================================================
# PHASE 2: BINARY TREE IN REDUCED SPACE
# ============================================================

def build_measurement_tree(Y_ops, V_supp, verbose=True):
    """
    Build binary tree from Kraus operators in the reduced (support) space.

    Args:
        Y_ops: (K, A_V, A_Vp) Kraus operators in full ortho bases
        V_supp: (A_Vp, n_support) support basis
        verbose: print diagnostics

    Returns:
        tree: BinaryKrausTree in reduced space
        B_targets: list of lists of (B_0, B_1) target pairs per level
    """
    K, A_V, A_Vp = Y_ops.shape
    n_support = V_supp.shape[1]

    # Project POVM elements to reduced space: M_k_red = V_supp† Y_k† Y_k V_supp
    # Equivalently: Y_k_red = Y_k @ V_supp, then M_k_red = Y_k_red† @ Y_k_red
    M_red = jnp.zeros((K, n_support, n_support), dtype=jnp.complex64)
    for k in range(K):
        Y_red = Y_ops[k] @ V_supp  # (A_V, n_support)
        M_red = M_red.at[k].set(dag(Y_red) @ Y_red)

    # Take sqrt(M_k_red) as Kraus operators for tree
    sqrt_M_red = jnp.zeros((K, n_support, n_support), dtype=jnp.complex64)
    for k in range(K):
        sqrt_M_red = sqrt_M_red.at[k].set(sqrt_psd(M_red[k]))

    # Build binary tree
    leaf_assign = jnp.arange(K, dtype=int)
    tree = BinaryKrausTree(sqrt_M_red, leaf_assign)

    if verbose:
        diff, comp = tree.check()
        print(f"  Tree: depth={tree.depth}, leaves={tree.M_count}, "
              f"reduced dim={n_support}")
        print(f"    Leaf POVM error: {diff:.2e}")
        print(f"    Completeness error: {comp:.2e}")

    # Extract B targets
    B_targets = []
    for level_nodes in tree.B_nodes:
        level_targets = []
        for (ba, bb) in level_nodes:
            level_targets.append((jnp.array(ba), jnp.array(bb)))
        B_targets.append(level_targets)

    return tree, B_targets


# ============================================================
# PHASE 3: CD+R OPTIMIZATION IN REDUCED SPACE (COHERENT BASIS)
# ============================================================

def project_circuit_to_support(alpha, beta, ds_prime, P_red):
    """
    Project CD+R Kraus operators to the E(P_C) support subspace.

    Computes K_red[mu] = P_red† K_coh[mu] P_red where:
        K_coh[mu][a,b] = sum_j alpha[mu,j] <d'_a|D(beta[mu,j])|d'_b>
        P_red = T_Vp @ V_supp  (B, n_support)

    Args:
        alpha: (2, N_l) complex coefficients from g()
        beta: (2, N_l) complex displacements from g()
        ds_prime: (B,) coherent state positions
        P_red: (B, n_support) reduced projection matrix

    Returns:
        K_red: (2, n_support, n_support) projected Kraus operators
    """
    B = ds_prime.shape[0]
    n_s = P_red.shape[1]
    N_l = alpha.shape[1]
    Pd = jnp.conj(P_red).T  # (n_support, B)

    K_red = jnp.zeros((2, n_s, n_s), dtype=jnp.complex64)

    for mu in range(2):
        # Displacement matrix elements: disp[j,a,b] = <d'_a|D(beta_j)|d'_b>
        # phase[j,b] = exp(-i*aOmegab(beta_j, d'_b))
        phase = jnp.exp(-1j * aOmegab(
            beta[mu, :, None], ds_prime[None, :]
        ))  # (N_l, B)
        # shifted[j,b] = beta_j + d'_b
        shifted = beta[mu, :, None] + ds_prime[None, :]  # (N_l, B)
        # overlap[a,j,b] = <d'_a | shifted_jb>
        ovlp = coherent_overlap(
            ds_prime[:, None, None],  # (B, 1, 1)
            shifted[None, :, :]       # (1, N_l, B)
        )  # (B, N_l, B)

        # K_coh[a,b] = sum_j alpha_j * phase[j,b] * ovlp[a,j,b]
        K_coh = jnp.einsum('j,jb,ajb->ab', alpha[mu], phase, ovlp)

        # Project: K_red = P_red† @ K_coh @ P_red
        K_red = K_red.at[mu].set(Pd @ K_coh @ P_red)

    return K_red


def procrustes_loss(K_red, B_0, B_1):
    """
    Procrustes-corrected loss: min_U ||K[0] - U B_0||² + ||K[1] - U B_1||²

    Analytically minimized over left-unitary U using SVD.
    """
    fixed = (jnp.sum(jnp.abs(K_red[0])**2) + jnp.sum(jnp.abs(B_0)**2)
             + jnp.sum(jnp.abs(K_red[1])**2) + jnp.sum(jnp.abs(B_1)**2))
    M = B_0 @ dag(K_red[0]) + B_1 @ dag(K_red[1])
    s = jnp.linalg.svdvals(M)
    return jnp.real(fixed - 2.0 * jnp.sum(s))


@partial(jax.jit, static_argnums=(3,))
def _compute_loss(params, ds_prime, P_red, N_l, B_0, B_1):
    """JIT-compiled loss for tree node optimization."""
    alpha, beta = g(params, N_l)
    K_red = project_circuit_to_support(alpha, beta, ds_prime, P_red)
    return procrustes_loss(K_red, B_0, B_1)


def optimize_tree_node(
    B_0, B_1, ds_prime, P_red,
    N_depth=6,
    lr=0.005,
    steps=3000,
    restarts=5,
    random_dist=3.0,
    random_angle=1.0,
    verbose=True,
):
    """
    Optimize CD+R circuit parameters for a single tree node.

    Minimizes Procrustes-corrected loss between projected circuit Kraus operators
    and tree node targets (B_0, B_1) in the reduced support space.

    Args:
        B_0, B_1: (n_support, n_support) target measurement operators
        ds_prime: (B,) coherent state positions
        P_red: (B, n_support) reduced projection matrix
        N_depth: CD+R layers (2^N_depth displacement terms)
        lr, steps, restarts: optimization parameters
        random_dist, random_angle: initialization scales
        verbose: print progress

    Returns:
        best_params: (N_depth, 4) optimized circuit parameters
        best_loss: final loss value
    """
    import sys
    N_l = 2 ** N_depth

    def loss_fn(params):
        return _compute_loss(params, ds_prime, P_red, N_l, B_0, B_1).real

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    best_loss = float('inf')
    best_params = None

    for restart in range(restarts):
        key = jr.PRNGKey(np.random.randint(100000))
        k1, k2, k3, key = jr.split(key, 4)

        params = jnp.zeros((N_depth, 4), jnp.complex64)
        params = params.at[:, 1:].set(
            2 * random_angle * jnp.pi * jr.uniform(key=k2, shape=(N_depth, 3))
        )
        params = params.at[:, 0].set(
            random_dist * jr.normal(key=k1, shape=(N_depth,))
            + random_dist * 1j * jr.normal(key=k3, shape=(N_depth,))
        )

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=lr * 0.1,
            peak_value=lr,
            warmup_steps=steps // 20,
            decay_steps=steps,
            end_value=lr * 0.01,
        )
        optimizer = optax.adam(schedule)
        opt_state = optimizer.init(params)

        for step in range(steps):
            loss, grads = grad_fn(params)
            updates, opt_state = optimizer.update(jnp.conj(grads), opt_state)
            params = optax.apply_updates(params, updates)
            params = params.at[:, 3].set(jnp.zeros(N_depth))

            if verbose and step % 500 == 0:
                print(f"    restart {restart}, step {step}: loss={float(loss):.6f}")
                sys.stdout.flush()

        final_loss = float(loss_fn(params))
        if verbose:
            print(f"    restart {restart} final: loss={final_loss:.6f}")
            sys.stdout.flush()

        if final_loss < best_loss:
            best_loss = final_loss
            best_params = jnp.array(params)
            if verbose:
                print(f"    >> New best! loss={best_loss:.6f}")
                sys.stdout.flush()

    return best_params, best_loss


def optimize_all_tree_nodes(
    B_targets, ds_prime, P_red,
    N_depth_sweep=(3, 4, 5, 6, 7, 8),
    lr=0.005,
    steps=3000,
    restarts=5,
    verbose=True,
):
    """
    Optimize CD+R circuits for all tree nodes across N_depth values.

    Args:
        B_targets: list of lists of (B_0, B_1) from build_measurement_tree
        ds_prime: (B,) coherent state positions
        P_red: (B, n_support) reduced projection matrix
        N_depth_sweep: N_depth values to sweep
        lr, steps, restarts: optimization parameters
        verbose: print progress

    Returns:
        results: dict mapping (level, node_idx, N_depth) -> {params, loss}
        best_per_node: dict mapping (level, node_idx) -> best result
    """
    import sys
    results = {}
    best_per_node = {}

    for level_idx, level_nodes in enumerate(B_targets):
        for node_idx, (B_0, B_1) in enumerate(level_nodes):
            node_key = (level_idx, node_idx)
            n_s = B_0.shape[0]
            if verbose:
                print(f"\n{'='*50}")
                print(f"  Level {level_idx}, Node {node_idx} "
                      f"(target dim: {n_s}x{n_s})")
                target_norms = (float(jnp.linalg.norm(B_0)),
                                float(jnp.linalg.norm(B_1)))
                print(f"  ||B_0||={target_norms[0]:.3f}, ||B_1||={target_norms[1]:.3f}")
                print(f"{'='*50}")
                sys.stdout.flush()

            node_best_loss = float('inf')
            node_best = None

            for N_depth in N_depth_sweep:
                N_l = 2 ** N_depth
                if verbose:
                    print(f"\n  N_depth={N_depth} (N_l={N_l})")
                    sys.stdout.flush()

                params, loss = optimize_tree_node(
                    B_0, B_1, ds_prime, P_red,
                    N_depth=N_depth, lr=lr, steps=steps,
                    restarts=restarts, verbose=verbose,
                )

                result_key = (level_idx, node_idx, N_depth)
                results[result_key] = {
                    'params': params, 'loss': loss, 'N_depth': N_depth
                }

                if loss < node_best_loss:
                    node_best_loss = loss
                    node_best = results[result_key]

            best_per_node[node_key] = node_best
            if verbose:
                print(f"\n  Best for node ({level_idx},{node_idx}): "
                      f"N_depth={node_best['N_depth']}, loss={node_best_loss:.6f}")
                sys.stdout.flush()

    return results, best_per_node


# ============================================================
# DISPLACEMENT COMPOSITION (for end-to-end optimization)
# ============================================================

def compose_single_displacement(alpha_A, beta_A, alpha_B, beta_B):
    """
    Compose K_A @ K_B in displacement representation.

    K_A = sum_j alpha_A[j] D(beta_A[j]),  K_B = sum_k alpha_B[k] D(beta_B[k])
    K_A @ K_B = sum_{j,k} alpha_A[j]*alpha_B[k]*exp(-i*aOmegab(beta_A[j],beta_B[k]))
                          * D(beta_A[j] + beta_B[k])

    Args:
        alpha_A, alpha_B: (N_A,), (N_B,) complex coefficients
        beta_A, beta_B: (N_A,), (N_B,) complex displacements

    Returns:
        alpha_AB: (N_A*N_B,) composed coefficients
        beta_AB: (N_A*N_B,) composed displacements
    """
    phase = e_n1iaOmegab(beta_A[:, None], beta_B[None, :])
    new_alpha = (alpha_A[:, None] * alpha_B[None, :] * phase).reshape(-1)
    new_beta = (beta_A[:, None] + beta_B[None, :]).reshape(-1)
    return new_alpha, new_beta


def build_leaf_displacements(all_params, tree_depth, N_l):
    """
    Build displacement representations for all tree leaves.

    Composes CD+R circuits along each leaf's tree path using the
    displacement composition rule: D(a)D(b) = exp(-i*aOmegab) D(a+b).

    Node ordering in all_params: level 0 has 1 node (idx 0),
    level 1 has 2 (idx 1,2), level d has 2^d (idx 2^d-1 .. 2^{d+1}-2).

    Args:
        all_params: (n_nodes, N_depth, 4) circuit parameters
        tree_depth: depth of binary tree
        N_l: 2^N_depth displacements per node outcome

    Returns:
        alpha_leaves: (n_leaves, N_l^tree_depth) complex coefficients
        beta_leaves: (n_leaves, N_l^tree_depth) complex displacements
    """
    n_leaves = 1 << tree_depth
    N_disp = N_l ** tree_depth
    n_nodes = (1 << tree_depth) - 1

    # Compute g() for each node
    node_alpha = []
    node_beta = []
    for i in range(n_nodes):
        a, b = g(all_params[i], N_l)
        node_alpha.append(a)  # (2, N_l)
        node_beta.append(b)   # (2, N_l)

    alpha_leaves = jnp.zeros((n_leaves, N_disp), dtype=jnp.complex64)
    beta_leaves = jnp.zeros((n_leaves, N_disp), dtype=jnp.complex64)

    for leaf in range(n_leaves):
        bits = [(leaf >> k) & 1 for k in range(tree_depth - 1, -1, -1)]
        parent_pos = 0

        # Start with root (level 0), outcome bits[0]
        a_eff = node_alpha[0][bits[0]]   # (N_l,)
        b_eff = node_beta[0][bits[0]]    # (N_l,)
        parent_pos = bits[0]

        # Compose remaining levels (outer node applied AFTER inner)
        for d in range(1, tree_depth):
            node_flat = (1 << d) - 1 + parent_pos
            a_node = node_alpha[node_flat][bits[d]]  # (N_l,)
            b_node = node_beta[node_flat][bits[d]]   # (N_l,)
            # K_node @ K_current
            a_eff, b_eff = compose_single_displacement(
                a_node, b_node, a_eff, b_eff
            )
            parent_pos = (parent_pos << 1) | bits[d]

        alpha_leaves = alpha_leaves.at[leaf].set(a_eff)
        beta_leaves = beta_leaves.at[leaf].set(b_eff)

    return alpha_leaves, beta_leaves


# ============================================================
# CMA-ES OPTIMIZATION (DERIVATIVE-FREE)
# ============================================================

def bipop_cmaes_flat(
    logical_0, logical_1, gamma,
    N_depth=6,
    n_restarts=10,
    popsize=80,
    maxiter=1000,
    sigma0=3.0,
    verbose=True,
):
    """
    BIPOP-style CMA-ES with multiple random restarts.

    The CD+R optimization landscape is highly multimodal: most CMA-ES seeds
    converge to the identity basin, but ~10-20% find significantly better
    recovery channels. This function runs many independent CMA-ES trials
    with different random seeds and returns the best result.

    Args:
        logical_0, logical_1: CoherentKet GKP logical states
        gamma: loss parameter
        N_depth: CD+R circuit depth (N_l = 2^N_depth)
        n_restarts: number of independent CMA-ES restarts
        popsize: CMA-ES population size
        maxiter: max generations per restart
        sigma0: initial step size
        verbose: print progress

    Returns:
        best_params: (N_depth, 4) optimized circuit parameters
        best_Fe: best entanglement fidelity
        info: dict with all trial results
    """
    import sys
    import cma
    import time

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
    def eval_circuit(p_complex):
        alpha, beta = g(p_complex, N_l)
        return entanglement_fidelity_displacement(
            alpha, beta,
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds, gamma)

    # JIT warmup
    _ = eval_circuit(jnp.zeros((N_depth, 4), dtype=jnp.complex64))

    def objective(x):
        return -float(eval_circuit(unpack(np.array(x))))

    Fe_id = -objective(np.zeros(N_depth * 4))

    if verbose:
        print(f"BIPOP CMA-ES: N_depth={N_depth}, N_l={N_l}, "
              f"restarts={n_restarts}, pop={popsize}")
        print(f"  Fe_id={Fe_id:.6f}")
        sys.stdout.flush()

    best_fe = Fe_id
    best_x = np.zeros(N_depth * 4)
    trials = []
    t_total = time.time()

    for trial in range(n_restarts):
        x0 = np.zeros(N_depth * 4)
        x0[0] = d_half; x0[3] = np.pi/2
        if N_depth > 1:
            x0[5] = d_half; x0[7] = np.pi/2

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

        fe = -es.result.fbest
        elapsed = time.time() - t0
        improved = fe > Fe_id + 0.001

        trials.append({'seed': trial, 'Fe': fe, 'gens': gen, 'time': elapsed})

        if verbose:
            flag = ' ***' if improved else ''
            print(f"  trial {trial:2d}: Fe={fe:.6f} "
                  f"({gen} gens, {elapsed:.0f}s){flag}")
            sys.stdout.flush()

        if fe > best_fe:
            best_fe = fe
            best_x = es.result.xbest.copy()

    elapsed_total = time.time() - t_total
    best_params = unpack(best_x)
    n_improved = sum(1 for t in trials if t['Fe'] > Fe_id + 0.001)

    if verbose:
        print(f"\n  Best Fe={best_fe:.6f} (+{best_fe-Fe_id:+.6f})")
        print(f"  Improved: {n_improved}/{n_restarts} trials")
        print(f"  Total time: {elapsed_total:.0f}s")
        sys.stdout.flush()

    return best_params, best_fe, {
        'Fe_id': Fe_id, 'trials': trials,
        'n_improved': n_improved,
        'total_time': elapsed_total,
    }


def optimize_cmaes_flat(
    logical_0, logical_1, gamma,
    N_depth=6,
    popsize=80,
    maxiter=2000,
    sigma0=3.0,
    seed=42,
    verbose=True,
):
    """
    CMA-ES optimization for flat (tree_depth=1, single node) recovery.

    Uses Covariance Matrix Adaptation Evolution Strategy, which is essential
    for escaping the identity basin that traps all gradient-based methods.

    The g() parameterization has 4 real params per layer: Re(d), Im(d), phi, theta.
    CMA-ES explores this space with a population of candidate solutions.

    Args:
        logical_0, logical_1: CoherentKet GKP logical states
        gamma: loss parameter
        N_depth: CD+R circuit depth (N_l = 2^N_depth displacements)
        popsize: CMA-ES population size (need >=80 for N_depth=6)
        maxiter: maximum CMA-ES generations
        sigma0: initial step size
        seed: random seed
        verbose: print progress

    Returns:
        best_params: (N_depth, 4) optimized circuit parameters
        best_Fe: best entanglement fidelity
        info: dict with optimization details
    """
    import sys
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
    def eval_circuit(p_complex):
        alpha, beta = g(p_complex, N_l)
        return entanglement_fidelity_displacement(
            alpha, beta,
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds, gamma)

    # JIT warmup
    _ = eval_circuit(jnp.zeros((N_depth, 4), dtype=jnp.complex64))

    def objective(x):
        return -float(eval_circuit(unpack(np.array(x))))

    # GKP-informed initial point
    x0 = np.zeros(n_params)
    x0[0] = d_half   # Re(d) for layer 0
    x0[3] = np.pi/2  # theta for layer 0 (balanced measurement)
    if N_depth > 1:
        x0[5] = d_half    # Re(d) for layer 1 (orthogonal direction)
        x0[7] = np.pi/2   # theta for layer 1

    Fe_id = float(eval_circuit(jnp.zeros((N_depth, 4), dtype=jnp.complex64)))

    if verbose:
        print(f"CMA-ES: N_depth={N_depth}, N_l={N_l}, params={n_params}, "
              f"pop={popsize}")
        print(f"  Fe_id={Fe_id:.6f}")
        sys.stdout.flush()

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'maxiter': maxiter, 'popsize': popsize,
        'verbose': -1, 'seed': seed, 'tolfun': 1e-9,
    })

    gen = 0
    best_ever = 0.0
    above_baseline_gen = None
    import time
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

    best_fe = -es.result.fbest
    elapsed = time.time() - t_start
    best_params = unpack(es.result.xbest)

    if verbose:
        print(f"  Done ({elapsed:.0f}s, {gen} gens): "
              f"Fe={best_fe:.6f}, improvement={best_fe-Fe_id:.6f}")
        sys.stdout.flush()

    return best_params, best_fe, {
        'Fe_id': Fe_id, 'generations': gen, 'elapsed': elapsed,
        'above_baseline_gen': above_baseline_gen,
        'xbest': es.result.xbest,
    }


def hybrid_cmaes_gradient(
    logical_0, logical_1, gamma,
    N_depth=6,
    popsize=80,
    cma_maxiter=2000,
    grad_steps=5000,
    grad_lr=0.001,
    sigma0=3.0,
    seed=42,
    verbose=True,
):
    """
    Hybrid CMA-ES + gradient descent optimization.

    Phase 1: CMA-ES explores the landscape to escape the identity basin
    Phase 2: Gradient descent fine-tunes the CMA-ES result

    This is the recommended approach: CMA-ES finds the basin of a good
    solution, then gradient descent converges precisely.

    Args:
        logical_0, logical_1: CoherentKet GKP logical states
        gamma: loss parameter
        N_depth: CD+R circuit depth
        popsize: CMA-ES population size
        cma_maxiter: max CMA-ES generations
        grad_steps: gradient descent steps
        grad_lr: peak learning rate for gradient descent
        sigma0: initial CMA-ES step size
        seed: random seed
        verbose: print progress

    Returns:
        best_params: (N_depth, 4) optimized circuit parameters
        best_Fe: best entanglement fidelity
        info: dict with optimization details
    """
    import sys
    import time

    N_l = 2 ** N_depth

    # Phase 1: CMA-ES
    if verbose:
        print(f"\n--- Phase 1: CMA-ES (N_depth={N_depth}) ---")
        sys.stdout.flush()

    params_cma, Fe_cma, cma_info = optimize_cmaes_flat(
        logical_0, logical_1, gamma,
        N_depth=N_depth, popsize=popsize,
        maxiter=cma_maxiter, sigma0=sigma0,
        seed=seed, verbose=verbose,
    )
    Fe_id = cma_info['Fe_id']

    # Phase 2: Gradient fine-tuning
    if verbose:
        print(f"\n--- Phase 2: Gradient fine-tuning ---")
        print(f"  Starting from CMA-ES Fe={Fe_cma:.6f}")
        sys.stdout.flush()

    def loss_fn(p):
        alpha, beta = g(p, N_l)
        Fe = entanglement_fidelity_displacement(
            alpha, beta,
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds, gamma)
        return (1.0 - Fe).real

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

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

    best_loss = float(loss_fn(params))
    best_params = jnp.array(params)
    t_start = time.time()

    for step in range(grad_steps):
        loss, grads = grad_fn(params)
        updates, opt_state = optimizer.update(jnp.conj(grads), opt_state)
        params = optax.apply_updates(params, updates)
        params = params.at[:, 3].set(jnp.zeros(N_depth))

        cur_loss = float(loss)
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_params = jnp.array(params)

        if verbose and step % 500 == 0:
            Fe = 1.0 - cur_loss
            elapsed = time.time() - t_start
            print(f"  step {step}: Fe={Fe:.6f} (best={1-best_loss:.6f}) "
                  f"[{elapsed:.0f}s]")
            sys.stdout.flush()

    Fe_final = 1.0 - best_loss
    elapsed = time.time() - t_start

    if verbose:
        print(f"  Fine-tuning done ({elapsed:.0f}s): Fe={Fe_final:.6f}")
        print(f"  CMA-ES: {Fe_cma:.6f} -> Fine-tuned: {Fe_final:.6f} "
              f"(delta={Fe_final-Fe_cma:+.6f})")
        print(f"  Improvement over identity: {Fe_final-Fe_id:+.6f}")
        sys.stdout.flush()

    return best_params, Fe_final, {
        **cma_info,
        'Fe_cma': Fe_cma,
        'Fe_final': Fe_final,
        'grad_steps': grad_steps,
    }


def optimize_cmaes_tree(
    logical_0, logical_1, gamma,
    tree_depth=2,
    N_depth=6,
    popsize=120,
    maxiter=3000,
    sigma0=3.0,
    seed=42,
    verbose=True,
):
    """
    CMA-ES optimization for tree recovery (tree_depth >= 2).

    Jointly optimizes all tree node circuits using displacement composition
    and CMA-ES. The parameter space is n_nodes * N_depth * 4.

    Args:
        logical_0, logical_1: CoherentKet GKP logical states
        gamma: loss parameter
        tree_depth: depth of binary tree
        N_depth: CD+R layers per tree node
        popsize: CMA-ES population size
        maxiter: max CMA-ES generations
        sigma0: initial CMA-ES step size
        seed: random seed
        verbose: print progress

    Returns:
        best_params: (n_nodes, N_depth, 4) optimized parameters
        best_Fe: best entanglement fidelity
        info: dict with optimization details
    """
    import sys
    import cma
    import time

    n_nodes = (1 << tree_depth) - 1
    N_l = 2 ** N_depth
    n_params = n_nodes * N_depth * 4
    d_half = float(jnp.sqrt(jnp.pi / 2))

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

    @jax.jit
    def eval_tree(p_complex):
        alpha_leaves, beta_leaves = build_leaf_displacements(
            p_complex, tree_depth, N_l)
        return entanglement_fidelity_displacement(
            alpha_leaves, beta_leaves,
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds, gamma)

    # JIT warmup
    _ = eval_tree(jnp.zeros((n_nodes, N_depth, 4), dtype=jnp.complex64))

    def objective(x):
        return -float(eval_tree(unpack(np.array(x))))

    # GKP-informed initial point for each node
    x0 = np.zeros(n_params)
    for node in range(n_nodes):
        base = node * N_depth * 4
        angle = node * np.pi / max(n_nodes, 1)
        x0[base] = d_half * np.cos(angle)
        x0[base+1] = d_half * np.sin(angle)
        x0[base+3] = np.pi/2

    Fe_id = -objective(np.zeros(n_params))

    if verbose:
        print(f"CMA-ES tree: depth={tree_depth}, n_nodes={n_nodes}, "
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

    best_fe = -es.result.fbest
    elapsed = time.time() - t_start
    best_params = unpack(es.result.xbest)

    if verbose:
        print(f"  Done ({elapsed:.0f}s, {gen} gens): "
              f"Fe={best_fe:.6f}, improvement={best_fe-Fe_id:.6f}")
        sys.stdout.flush()

    return best_params, best_fe, {
        'Fe_id': Fe_id, 'generations': gen, 'elapsed': elapsed,
        'above_baseline_gen': above_baseline_gen,
        'n_nodes': n_nodes, 'tree_depth': tree_depth,
    }


# ============================================================
# ASSEMBLY AND EVALUATION (per-node approach)
# ============================================================

def assemble_tree_recovery(tree, best_per_node, ds_prime, P_red):
    """
    Assemble full recovery from optimized tree node circuits.

    For each leaf, compose CD+R Kraus operators along the tree path,
    all in the reduced support space.

    Args:
        tree: BinaryKrausTree
        best_per_node: dict from optimize_all_tree_nodes
        ds_prime: (B,) coherent state positions
        P_red: (B, n_support) reduced projection

    Returns:
        leaf_ops: (M, n_support, n_support) effective leaf Kraus operators
    """
    n_s = P_red.shape[1]
    M = tree.M_count
    depth = tree.depth

    # Pre-compute K_red for each node
    node_K = {}
    for level_idx in range(depth):
        for node_idx in range(2 ** level_idx):
            result = best_per_node[(level_idx, node_idx)]
            N_l = 2 ** result['N_depth']
            alpha, beta = g(result['params'], N_l)
            K_red = project_circuit_to_support(alpha, beta, ds_prime, P_red)
            node_K[(level_idx, node_idx)] = K_red

    leaf_ops = jnp.zeros((M, n_s, n_s), dtype=jnp.complex64)

    for leaf in range(M):
        bits = [(leaf >> k) & 1 for k in range(depth - 1, -1, -1)]
        parent_index = 0
        K_eff = jnp.eye(n_s, dtype=jnp.complex64)

        for d, bit in enumerate(bits):
            K_node = node_K[(d, parent_index)]
            K_eff = K_node[bit] @ K_eff
            parent_index = (parent_index << 1) | bit

        leaf_ops = leaf_ops.at[leaf].set(K_eff)

    return leaf_ops


def evaluate_fidelity_fock(leaf_ops, V_supp, subspace_Vp, logical_0, logical_1,
                           gamma, loss_rank=10, N=GKP_N):
    """
    Evaluate entanglement fidelity by synthesizing to Fock basis.

    Maps the reduced-space leaf operators back to full Hilbert space via V_supp.

    Args:
        leaf_ops: (M, n_support, n_support) recovery ops in reduced space
        V_supp: (A_Vp, n_support) support basis in V' ortho
        subspace_Vp: BosonicSubspace for V'
        logical_0, logical_1: CoherentKet GKP logical states
        gamma, loss_rank, N: Fock-basis parameters

    Returns:
        Fe: entanglement fidelity
    """
    M, n_s, _ = leaf_ops.shape

    # Lift leaf ops to full V' ortho: Y_full = V_supp @ leaf_ops @ V_supp†
    leaf_full = jnp.zeros((M, V_supp.shape[0], V_supp.shape[0]), dtype=jnp.complex64)
    for k in range(M):
        leaf_full = leaf_full.at[k].set(V_supp @ leaf_ops[k] @ dag(V_supp))

    A_Vp = V_supp.shape[0]

    # Build Fock kets for V' ortho frame
    fock_Vp = jnp.squeeze(
        jax.vmap(lambda alpha: dqcoherent(N, alpha))(subspace_Vp.ds)
    )
    if fock_Vp.ndim == 3:
        fock_Vp = fock_Vp.squeeze(-1)

    phi_Vp = jnp.einsum('ba,bn->an', subspace_Vp.T, fock_Vp)  # (A_Vp, N)

    # Build Fock recovery operators
    recovery_fock = jnp.zeros((M, N, N), dtype=jnp.complex64)
    for k in range(M):
        R_k = jnp.einsum('ij,in,jm->nm', leaf_full[k], phi_Vp, jnp.conj(phi_Vp))
        recovery_fock = recovery_fock.at[k].set(R_k)

    # Build Fock logical states
    A0 = logical_0.cs.shape[0]
    all_ds = jnp.concatenate([logical_0.ds, logical_1.ds])
    fock_all = jnp.squeeze(
        jax.vmap(lambda alpha: dqcoherent(N, alpha))(all_ds)
    )
    if fock_all.ndim == 3:
        fock_all = fock_all.squeeze(-1)

    psi_0 = jnp.einsum('bn,b->n', fock_all[:A0], logical_0.cs)
    psi_1 = jnp.einsum('bn,b->n', fock_all[A0:], logical_1.cs)
    psi_0 = psi_0 / jnp.sqrt(jnp.real(jnp.dot(jnp.conj(psi_0), psi_0)))
    psi_1 = psi_1 / jnp.sqrt(jnp.real(jnp.dot(jnp.conj(psi_1), psi_1)))
    psi_0 = psi_0.reshape(-1, 1)
    psi_1 = psi_1.reshape(-1, 1)

    loss_ops = make_pureloss_fock(gamma, rank=loss_rank, N=N)

    psi = [psi_0, psi_1]
    Fe = 0.0
    for mu in range(2):
        for nu in range(2):
            rho_mn = psi[mu] @ dqdag(psi[nu])
            after_loss = apply_kraus_map_nonorm(loss_ops, rho_mn)
            after_recovery = apply_kraus_map_nonorm(recovery_fock, after_loss)
            Fe += (dqdag(psi[mu]) @ after_recovery @ psi[nu]).squeeze()
    Fe = float(jnp.real(Fe) / 4.0)

    return Fe


# ============================================================
# ANALYTIC ENTANGLEMENT FIDELITY (COHERENT BASIS)
# ============================================================

def entanglement_fidelity_displacement(alpha, beta, c_0, d_0, c_1, d_1, gamma):
    """
    Compute entanglement fidelity directly in the coherent basis.

    Fe = (1/4) sum_{mu,nu} sum_k <psi_mu| R_k E(|psi_mu><psi_nu|) R_k† |psi_nu>

    where R_k = sum_j alpha[k,j] D(beta[k,j]) are recovery operators and
    E is the pure loss channel with parameter gamma.

    This is much faster than the general-purpose
    analytic_pureloss_recovery_fidelity_thetaphi because it avoids
    triple-nested vmaps by computing the key inner products via einsum.

    Args:
        alpha: (n_kraus, N_disp) complex Kraus coefficients
        beta: (n_kraus, N_disp) complex displacement positions
        c_0: (A0,) logical |0> coefficients
        d_0: (A0,) logical |0> positions
        c_1: (A1,) logical |1> coefficients
        d_1: (A1,) logical |1> positions
        gamma: loss parameter

    Returns:
        Fe: entanglement fidelity (real scalar)
    """
    t = jnp.sqrt(1.0 - gamma)
    r = jnp.sqrt(gamma)
    n_kraus = alpha.shape[0]

    cs = [c_0, c_1]
    ds = [d_0, d_1]

    # Precompute env overlaps for all (mu,nu) pairs
    # env[mu][nu][b,a] = <r*d_nu[b] | r*d_mu[a]>
    env = {}
    for mu in range(2):
        env[mu] = {}
        for nu in range(2):
            env[mu][nu] = coherent_overlap(
                r * ds[nu].reshape(-1, 1),   # (A_nu, 1)
                r * ds[mu].reshape(1, -1),   # (1, A_mu)
            )  # (A_nu, A_mu): env[mu][nu][b,a] = <r*d_nu[b]|r*d_mu[a]>

    Fe = 0.0 + 0j

    for k in range(n_kraus):
        # For each Kraus operator, compute L_k^{mu}[a] = <psi_mu| R_k |t*d_mu[a]>
        L = {}
        for mu in range(2):
            A_mu = ds[mu].shape[0]
            td_mu = t * ds[mu]  # (A_mu,)

            # phase[j,a] = exp(-i * aOmegab(beta[k,j], t*d_mu[a]))
            phase = jnp.exp(-1j * aOmegab(
                beta[k, :, None],    # (N_disp, 1)
                td_mu[None, :],      # (1, A_mu)
            ))  # (N_disp, A_mu)

            # shifted[j,a] = beta[k,j] + t*d_mu[a]
            shifted = beta[k, :, None] + td_mu[None, :]  # (N_disp, A_mu)

            # ovlp[p,j,a] = <d_mu[p] | shifted[j,a]> = <d_mu[p] | beta_kj + t*d_mu[a]>
            ovlp = coherent_overlap(
                ds[mu][:, None, None],   # (A_mu, 1, 1)
                shifted[None, :, :],     # (1, N_disp, A_mu)
            )  # (A_mu, N_disp, A_mu)

            # L_k^{mu}[a] = sum_{p,j} conj(c_mu[p]) * alpha[k,j] * phase[j,a] * ovlp[p,j,a]
            L[mu] = jnp.einsum(
                'p,j,ja,pja->a',
                jnp.conj(cs[mu]), alpha[k], phase, ovlp,
            )

        # Accumulate Fe contributions for all (mu,nu)
        for mu in range(2):
            for nu in range(2):
                v_mu = cs[mu] * L[mu]   # (A_mu,)
                v_nu = cs[nu] * L[nu]   # (A_nu,)
                # contribution = sum_{a,b} v_mu[a] * conj(v_nu[b]) * env[mu][nu][b,a]
                #              = conj(v_nu) @ env[mu][nu] @ v_mu
                Fe += jnp.conj(v_nu) @ env[mu][nu] @ v_mu

    return jnp.real(Fe) / 4.0


# ============================================================
# INITIALIZATION STRATEGIES
# ============================================================

def gkp_informed_init(n_nodes, N_depth, key, lattice="square",
                      perturbation_scale=0.5):
    """
    Physics-informed initialization for tree CD+R circuits.

    Sets layer 0 of each node to a GKP-syndrome-detecting displacement
    with balanced qubit rotation. Remaining layers get progressively
    smaller displacements. Different nodes probe different lattice directions.

    Args:
        n_nodes: number of tree nodes
        N_depth: CD+R layers per node
        key: JAX random key
        lattice: "square" or "hexagonal"
        perturbation_scale: scale of random perturbation on remaining layers

    Returns:
        params: (n_nodes, N_depth, 4) initial parameters
    """
    if lattice == "square":
        d_syndrome = jnp.sqrt(jnp.pi / 2)  # Half-period of square lattice
    else:
        d_syndrome = jnp.sqrt(jnp.pi / jnp.sqrt(3))

    params = jnp.zeros((n_nodes, N_depth, 4), jnp.complex64)
    k1, k2, k3, key = jr.split(key, 4)

    for i in range(n_nodes):
        # Layer 0: Main syndrome displacement + balanced measurement
        # Rotate phase per node to probe different lattice directions
        angle = i * jnp.pi / max(n_nodes, 1)
        d = d_syndrome * jnp.exp(1j * angle)
        params = params.at[i, 0, 0].set(d)
        params = params.at[i, 0, 1].set(angle)          # phi
        params = params.at[i, 0, 2].set(jnp.pi / 2)    # theta = pi/2 (balanced)

        # Remaining layers: progressively smaller displacements
        for layer in range(1, N_depth):
            scale = d_syndrome / (2.0 ** layer)
            params = params.at[i, layer, 0].set(
                scale * (jr.normal(jr.fold_in(k1, i * N_depth + layer), ())
                         + 1j * jr.normal(jr.fold_in(k2, i * N_depth + layer), ()))
            )
            params = params.at[i, layer, 1].set(
                perturbation_scale * jnp.pi
                * jr.uniform(jr.fold_in(k3, i * N_depth + layer), ())
            )
            params = params.at[i, layer, 2].set(
                perturbation_scale * jnp.pi
                * jr.uniform(jr.fold_in(key, i * N_depth + layer), ())
            )

    return params


def procrustes_warm_start(transpose_result, tree_depth, N_depth,
                          steps=1500, restarts=2, verbose=True):
    """
    Compute warm-start params by per-node Procrustes optimization.

    Runs the binary tree decomposition of the transpose channel and
    optimizes each node's CD+R circuit to match the tree targets.

    Args:
        transpose_result: dict from build_coherent_transpose_channel
        tree_depth: depth of binary tree
        N_depth: CD+R layers per node
        steps, restarts: optimization parameters
        verbose: print progress

    Returns:
        init_params: (n_nodes, N_depth, 4) warm-start parameters
    """
    import sys

    # Phase 2: Build tree
    V_supp, n_support, P_red = compute_support_basis(transpose_result)
    Y_ops = transpose_result['Y_ops']
    subspace_Vp = transpose_result['subspace_Vp']
    ds_prime = transpose_result['ds_all']

    tree, B_targets = build_measurement_tree(Y_ops, V_supp, verbose=verbose)

    # Truncate tree to desired depth
    B_targets_trunc = B_targets[:tree_depth]

    # Phase 3: Per-node optimization
    n_nodes = (1 << tree_depth) - 1
    init_params = jnp.zeros((n_nodes, N_depth, 4), jnp.complex64)

    node_idx = 0
    for level_idx, level_nodes in enumerate(B_targets_trunc):
        for nidx, (B_0, B_1) in enumerate(level_nodes):
            if verbose:
                print(f"  Warm-start: node ({level_idx},{nidx}), "
                      f"N_depth={N_depth}")
                sys.stdout.flush()

            params, loss = optimize_tree_node(
                B_0, B_1, ds_prime, P_red,
                N_depth=N_depth, lr=0.005, steps=steps,
                restarts=restarts, verbose=verbose,
            )
            init_params = init_params.at[node_idx].set(params)
            node_idx += 1

    return init_params


# ============================================================
# END-TO-END TREE OPTIMIZATION (DISPLACEMENT COMPOSITION)
# ============================================================

def optimize_tree_end_to_end(
    logical_0, logical_1, gamma,
    tree_depth=2,
    N_depth=5,
    lr=0.003,
    steps=5000,
    restarts=5,
    random_dist=4.0,
    random_angle=1.0,
    init_params=None,
    verbose=True,
):
    """
    End-to-end tree optimization using analytic fidelity with displacement
    composition.

    Composes all tree node circuits into effective leaf displacement operators
    using D(a)D(b) = exp(-i*aOmegab)*D(a+b), then evaluates entanglement
    fidelity directly. This avoids the support-space projection and
    guarantees physically valid fidelity values (Fe <= 1).

    Uses three initialization strategies:
      1. GKP-informed init (first restart) - physics-based syndrome displacements
      2. Warm-start from init_params if provided (second restart)
      3. Random initialization (remaining restarts)

    Args:
        logical_0, logical_1: CoherentKet GKP logical states
        gamma: loss parameter
        tree_depth: depth of binary tree (2^depth leaves)
        N_depth: CD+R layers per tree node
        lr, steps, restarts: optimization parameters
        random_dist, random_angle: initialization scales
        init_params: optional (n_nodes, N_depth, 4) warm-start params
        verbose: print progress

    Returns:
        best_params: (n_nodes, N_depth, 4) optimized parameters
        best_Fe: best entanglement fidelity
    """
    import sys

    n_nodes = (1 << tree_depth) - 1
    n_leaves = 1 << tree_depth
    N_l = 2 ** N_depth
    N_disp_leaf = N_l ** tree_depth

    if verbose:
        print(f"  End-to-end tree optimization (displacement composition):")
        print(f"    tree_depth={tree_depth}, n_nodes={n_nodes}, "
              f"n_leaves={n_leaves}")
        print(f"    N_depth={N_depth}, N_l={N_l}, "
              f"N_disp/leaf={N_disp_leaf}")
        print(f"    Total params: {n_nodes * N_depth * 4}")
        init_modes = ["GKP-informed"]
        if init_params is not None:
            init_modes.append("warm-start")
        init_modes.append(f"random x{max(0, restarts - len(init_modes))}")
        print(f"    Init strategies: {', '.join(init_modes)}")
        sys.stdout.flush()

    def loss_fn(all_params):
        alpha_leaves, beta_leaves = build_leaf_displacements(
            all_params, tree_depth, N_l
        )
        Fe = entanglement_fidelity_displacement(
            alpha_leaves, beta_leaves,
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds,
            gamma,
        )
        return (1.0 - Fe).real

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    eval_fn = jax.jit(loss_fn)

    best_loss = 1.0
    best_params = None

    for restart in range(restarts):
        key = jr.PRNGKey(np.random.randint(100000))

        # Choose initialization strategy
        if restart == 0:
            # GKP-informed initialization
            params = gkp_informed_init(n_nodes, N_depth, key)
            init_label = "GKP-informed"
        elif restart == 1 and init_params is not None:
            # Warm-start from provided params
            params = jnp.array(init_params, dtype=jnp.complex64)
            init_label = "warm-start"
        else:
            # Random initialization
            k1, k2, k3, key = jr.split(key, 4)
            params = jnp.zeros((n_nodes, N_depth, 4), jnp.complex64)
            params = params.at[:, :, 1:].set(
                2 * random_angle * jnp.pi
                * jr.uniform(key=k2, shape=(n_nodes, N_depth, 3))
            )
            params = params.at[:, :, 0].set(
                random_dist * jr.normal(key=k1, shape=(n_nodes, N_depth))
                + random_dist * 1j * jr.normal(
                    key=k3, shape=(n_nodes, N_depth))
            )
            init_label = "random"

        if verbose:
            print(f"    restart {restart} ({init_label}):")
            sys.stdout.flush()

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=lr * 0.1,
            peak_value=lr,
            warmup_steps=steps // 20,
            decay_steps=steps,
            end_value=lr * 0.01,
        )
        optimizer = optax.adam(schedule)
        opt_state = optimizer.init(params)

        for step in range(steps):
            loss, grads = grad_fn(params)
            updates, opt_state = optimizer.update(jnp.conj(grads), opt_state)
            params = optax.apply_updates(params, updates)
            params = params.at[:, :, 3].set(
                jnp.zeros((n_nodes, N_depth))
            )

            if verbose and step % 500 == 0:
                Fe = 1.0 - float(loss)
                print(f"      step {step}: Fe={Fe:.6f}")
                sys.stdout.flush()

        final_loss = float(eval_fn(params))
        Fe = 1.0 - final_loss
        if verbose:
            print(f"      final: Fe={Fe:.6f}")
            sys.stdout.flush()

        if final_loss < best_loss:
            best_loss = final_loss
            best_params = jnp.array(params)
            if verbose:
                print(f"      >> New best! Fe={Fe:.6f}")
                sys.stdout.flush()

    best_Fe = 1.0 - best_loss
    return best_params, best_Fe


def evaluate_tree_fidelity_fock(all_params, tree_depth, N_l,
                                 logical_0, logical_1, gamma,
                                 loss_rank=10, N=GKP_N):
    """
    Evaluate tree recovery entanglement fidelity in Fock basis.

    Composes tree node circuits via displacement composition to get leaf
    Kraus operators, synthesizes them in Fock basis, then computes Fe.

    Args:
        all_params: (n_nodes, N_depth, 4) circuit parameters
        tree_depth, N_l: tree structure parameters
        logical_0, logical_1: CoherentKet GKP logical states
        gamma: loss parameter
        loss_rank: number of loss Kraus operators
        N: Fock space dimension

    Returns:
        Fe: entanglement fidelity (scalar)
    """
    alpha_leaves, beta_leaves = build_leaf_displacements(
        all_params, tree_depth, N_l
    )

    # Synthesize recovery operators in Fock basis
    recovery_ops = channel_from_b(alpha_leaves, beta_leaves)

    # Build Fock-basis loss operators
    loss_ops = make_pureloss_fock(gamma, rank=loss_rank, N=N)

    # Build Fock logical states
    fock_states = []
    for ck in [logical_0, logical_1]:
        coherents = jnp.squeeze(
            jax.vmap(lambda alpha: dqcoherent(N, alpha))(ck.ds)
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
            rho_mn = fock_states[mu] @ dag(fock_states[nu])
            after_loss = apply_kraus_map_nonorm(loss_ops, rho_mn)
            after_recovery = apply_kraus_map_nonorm(recovery_ops, after_loss)
            Fe += (dag(fock_states[mu]) @ after_recovery @ fock_states[nu]).squeeze()
    Fe = float(jnp.real(Fe) / 4.0)
    return Fe


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_tree_pipeline(
    transpose_result,
    tree_depth=2,
    N_depth=5,
    lr=0.003,
    steps=3000,
    restarts=3,
    n_bloch_points=32,
    verbose=True,
):
    """
    Full pipeline: end-to-end tree optimization + Fock cross-validation.

    Uses displacement composition to jointly optimize all tree node circuits
    for maximum average fidelity.

    Args:
        transpose_result: dict from build_coherent_transpose_channel
        tree_depth: depth of binary tree (2^depth leaves)
        N_depth: CD+R layers per tree node
        lr, steps, restarts: optimization hyperparameters
        n_bloch_points: Bloch sphere samples for fidelity averaging
        verbose: print progress

    Returns:
        pipeline_result: dict with optimization results and fidelity
    """
    import sys

    gamma = transpose_result['gamma']
    logical_0 = transpose_result['logical_0']
    logical_1 = transpose_result['logical_1']
    N_l = 2 ** N_depth
    n_leaves = 1 << tree_depth

    if verbose:
        print(f"\n{'='*60}")
        print(f"Tree Pipeline: gamma={gamma}, depth={tree_depth}, "
              f"N_depth={N_depth}")
        print(f"  Leaves={n_leaves}, N_l={N_l}, "
              f"N_disp/leaf={N_l**tree_depth}")
        print(f"{'='*60}")
        sys.stdout.flush()

    # End-to-end optimization
    if verbose:
        print("\n--- End-to-End Optimization ---")
        sys.stdout.flush()

    best_params, best_Fe = optimize_tree_end_to_end(
        logical_0, logical_1, gamma,
        tree_depth=tree_depth, N_depth=N_depth,
        lr=lr, steps=steps, restarts=restarts,
        verbose=verbose,
    )

    # Fock cross-validation
    if verbose:
        print("\n--- Fock Cross-Validation ---")
        sys.stdout.flush()

    Fe_fock = evaluate_tree_fidelity_fock(
        best_params, tree_depth, N_l,
        logical_0, logical_1, gamma,
    )

    Fe_transpose = transpose_result.get('Fe_fock_transpose', None)
    Fe_none = transpose_result.get('Fe_none', None)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Results for gamma={gamma}")
        print(f"{'='*60}")
        print(f"  F_e (tree, analytic est): {best_Fe:.6f}")
        print(f"  F_e (tree, Fock):         {Fe_fock:.6f}")
        if Fe_transpose is not None:
            print(f"  F_e (transpose bound):    {Fe_transpose:.6f}")
            print(f"  Gap to bound:             {Fe_transpose - Fe_fock:.6f}")
        if Fe_none is not None:
            print(f"  F_e (no recovery):        {Fe_none:.6f}")
            print(f"  Improvement:              {Fe_fock - Fe_none:.6f}")
        sys.stdout.flush()

    return {
        'gamma': gamma,
        'tree_depth': tree_depth,
        'N_depth': N_depth,
        'n_leaves': n_leaves,
        'best_params': best_params,
        'Fe_analytic': best_Fe,
        'Fe_fock': Fe_fock,
        'Fe_transpose': Fe_transpose,
        'Fe_none': Fe_none,
    }


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    import sys
    from coherax.characteristic_jax_utils import gkp_coherent_dm

    print("=" * 60)
    print("Coherent Tree Optimizer - CMA-ES + Gradient Hybrid")
    print("=" * 60)

    gamma = 0.1
    logical_0 = gkp_coherent_dm(mu=0, N_trunc=3, Delta=0.3, lattice='square')
    logical_1 = gkp_coherent_dm(mu=1, N_trunc=3, Delta=0.3, lattice='square')

    # Compute baseline
    Fe_id = float(entanglement_fidelity_displacement(
        jnp.ones((1, 1), dtype=jnp.complex64),
        jnp.zeros((1, 1), dtype=jnp.complex64),
        logical_0.cs, logical_0.ds,
        logical_1.cs, logical_1.ds, gamma))
    print(f"gamma={gamma}, Fe_id={Fe_id:.6f}")
    sys.stdout.flush()

    # --- Flat CMA-ES sweep across N_depth ---
    print(f"\n{'='*60}")
    print("Flat CMA-ES sweep (tree_depth=1)")
    print(f"{'='*60}")

    flat_results = {}
    for Nd in [5, 6, 7]:
        pop = 80 if Nd <= 6 else 200
        params, fe, info = optimize_cmaes_flat(
            logical_0, logical_1, gamma,
            N_depth=Nd, popsize=pop, maxiter=2000,
            verbose=True,
        )
        flat_results[Nd] = (params, fe, info)

    # Gradient fine-tune the best flat result
    best_Nd = max(flat_results.keys(), key=lambda k: flat_results[k][1])
    best_fe_cma = flat_results[best_Nd][1]

    if best_fe_cma > Fe_id + 0.001:
        print(f"\n--- Gradient fine-tuning best flat (N_depth={best_Nd}) ---")
        params_ft, Fe_ft, ft_info = hybrid_cmaes_gradient(
            logical_0, logical_1, gamma,
            N_depth=best_Nd, popsize=80, cma_maxiter=2000,
            grad_steps=5000, verbose=True,
        )

        # Fock cross-validation
        print("\n--- Fock Cross-Validation ---")
        N_l = 2 ** best_Nd
        # For flat recovery, wrap params as (1, N_depth, 4) tree with depth=1
        # Actually just use evaluate with tree_depth=1
        alpha, beta = g(params_ft, N_l)
        Fe_fock = float(entanglement_fidelity_displacement(
            alpha, beta,
            logical_0.cs, logical_0.ds,
            logical_1.cs, logical_1.ds, gamma))

        print(f"\n{'='*60}")
        print(f"  Results (gamma={gamma})")
        print(f"{'='*60}")
        print(f"  Fe (identity):      {Fe_id:.6f}")
        print(f"  Fe (CMA-ES):        {best_fe_cma:.6f}")
        print(f"  Fe (fine-tuned):    {Fe_ft:.6f}")
        print(f"  Improvement:        {Fe_ft - Fe_id:+.6f}")
    else:
        print(f"\nNo improvement found over identity.")
