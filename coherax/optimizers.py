"""Gradient-based optimization tools that leverage the coherent-basis native tools."""

from __future__ import annotations

import time
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jaxtyping import Array

from coherax.linalg_utils import complex_normal
from coherax.states import CoherentKet
from coherax.fidelity import (
    analytic_fidelity_wrapper,
    analytic_fidelity_transfer_wrapper,
    coherent_information_pureloss,
    entanglement_fidelity_pureloss,
    nbar_logical,
)


def optimize_ECD_state_prep(
    target_state: CoherentKet,
    N_depth=6,
    lr=0.005,
    steps=10000,
    restarts=5,
    random_dist=0.1,
    random_angle=jnp.pi,
    initial=None,
):
    N_layer = 2 ** (N_depth)

    @partial(jax.jit, static_argnums=0)
    def analytic_loss_fn(N_l: int, circuit_params: Array):
        return 1.0 - analytic_fidelity_wrapper(target_state, circuit_params, N_l)

    @partial(jax.jit, static_argnums=2)
    def analytic_train_step(a: Array, opt_state: Any, N_l: int):
        a = a.astype(jnp.complex64)
        partial_analytic_loss_fn = partial(analytic_loss_fn, N_l)
        grads = jax.grad(partial_analytic_loss_fn)(a)
        grads = jnp.conj(grads)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        a_updated = optax.apply_updates(a, updates)
        return a_updated, new_opt_state

    best_loss = 1.0
    best_val = None
    loss_vals = jnp.zeros((steps,), dtype=jnp.float32)
    for restart in range(restarts):
        key = jr.PRNGKey(np.random.randint(1000))
        optimizer = optax.adam(lr)
        k1, k2, k3 = jr.split(key, 3)
        a_init = jnp.zeros((N_depth, 4), jnp.complex64)
        if initial is not None:
            a_init = initial
        a_init = a_init.at[:, 1:3].add(
            2 * random_angle * jr.uniform(key=k2, shape=(N_depth, 2))
        )
        a_init = a_init.at[:, 0].add(
            random_dist * jr.normal(key=k1, shape=(N_depth,))
            + random_dist * 1.0j * jr.normal(key=k3, shape=(N_depth,))
        )
        a = a_init

        # print(a)
        opt_state = optimizer.init(a_init)
        last_loss = 1.0
        for step_i in range(steps):
            a, opt_state = analytic_train_step(a=a, opt_state=opt_state, N_l=N_layer)
            a = a.at[
                :, 3
            ].set(
                jnp.zeros(
                    N_depth,
                )
            )  # kill the gammas, this is sort of optional but will give the simplest circuit
            if step_i % 100 == 0:
                current_loss = analytic_loss_fn(N_layer, a)
                loss_vals = loss_vals.at[step_i].set(current_loss.item())
                print(
                    f"Restart {restart}, Step {step_i}, 1 - F = {current_loss.item():.6f}"
                )
                if last_loss == current_loss and current_loss > 0.05:
                    print(f"Ending restart {restart} early.")
                    # a = a.at[:, 1:].set(0.01 * jr.normal(key=k2, shape=(N_depth, 3)))
                    break
                last_loss = current_loss
        current_loss = analytic_loss_fn(N_l=N_layer, circuit_params=a)
        if current_loss < best_loss:
            best_loss = current_loss
            best_val = a
            print(restart, "new best", best_loss)
            print(best_val)

    return best_val, best_loss


def optimize_ECD_state_transfer(
    start_state: CoherentKet,
    final_state: CoherentKet,
    T_depth=1,
    N_depth=6,
    lr=0.005,
    steps=10000,
    restarts=5,
    random_dist=1.0,
    random_angle=jnp.pi,
    initial=None,
):
    N_layer = 2 ** (N_depth)

    @jax.jit
    def analytic_fidelity_transfer_loss_fn(circuit_params: Array):
        return 1.0 - analytic_fidelity_transfer_wrapper(
            initial=start_state,
            final=final_state,
            circuit_params=circuit_params,
            N_l=N_layer,
            T=T_depth,
        )

    def analytic_train_transfer_step(a: Array, opt_state: Any):
        a = a.astype(jnp.complex64)
        grads = jax.grad(analytic_fidelity_transfer_loss_fn)(a)
        grads = jnp.conj(grads)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        a_updated = optax.apply_updates(a, updates)
        return a_updated, new_opt_state

    best_loss = 1.0
    best_val = None
    loss_vals = jnp.zeros((steps,), dtype=jnp.float32)
    for restart in range(restarts):
        key = jr.PRNGKey(np.random.randint(1000))
        optimizer = optax.adam(lr)
        k1, k2, k3 = jr.split(key, 3)
        a_init = jnp.zeros((T_depth, N_depth, 4), jnp.complex64)
        if initial is not None:
            a_init = initial
        a_init = a_init.at[:, :, 1:].add(
            2 * random_angle * jr.uniform(key=k2, shape=(T_depth, N_depth, 3))
        )
        a_init = a_init.at[:, :, 0].add(
            random_dist
            * jr.normal(
                key=k1,
                shape=(
                    T_depth,
                    N_depth,
                ),
            )
            + random_dist
            * 1.0j
            * jr.normal(
                key=k3,
                shape=(
                    T_depth,
                    N_depth,
                ),
            )
        )
        a = a_init
        print(a)
        opt_state = optimizer.init(a_init)
        last_loss = 1.0
        for step_i in range(steps):
            a, opt_state = analytic_train_transfer_step(a=a, opt_state=opt_state)
            a = a.at[:, :, 3].set(jnp.zeros((T_depth, N_depth)))
            if step_i % 100 == 0:
                current_loss = analytic_fidelity_transfer_loss_fn(a)
                loss_vals = loss_vals.at[step_i].set(current_loss.item())
                print(
                    f"Restart {restart}, Step {step_i}, 1 - F = {current_loss.item():.6f}"
                )
                if last_loss == current_loss and current_loss > 0.05:
                    print(f"Ending restart {restart} early.")
                    break
                last_loss = current_loss
        current_loss = analytic_fidelity_transfer_loss_fn(a)
        if current_loss < best_loss:
            best_loss = current_loss
            best_val = a
            print(restart, "new best", best_loss)
            print(best_val)

    return best_val, best_loss


# ---------------------------------------------------------------------------
# Floating-basis encoder / decoder optimization
#
# Two-phase Adam restart loops for maximizing F_e (joint encoder + Kraus
# decoder) and I_c (encoder only). The objectives are
# :func:`~coherax.fidelity.entanglement_fidelity_pureloss` and
# :func:`~coherax.fidelity.coherent_information_pureloss`; the optimizer
# adds a soft separation penalty on the coherent positions to keep the
# Gram matrix well-conditioned during exploration.
# ---------------------------------------------------------------------------


def init_separated_d(
    key: Array,
    N_C: int,
    min_sep: float = 1.0,
    scale: float = 2.0,
) -> Array:
    r"""Place ``N_C`` coherent-state positions on concentric rings.

    Distributes ``N_C`` points on one or two rings so that all pairwise
    distances are :math:`\geq \min_{\mathrm{sep}}`, then adds a small
    complex perturbation. Designed to keep the Gram matrix
    well-conditioned at the start of a gradient-descent run.

    Parameters
    ----------
    key : jax.random.PRNGKey
        For the perturbation.
    N_C : int
        Number of points.
    min_sep : float
        Target minimum pairwise distance.
    scale : float
        Overall radius scale.

    Returns
    -------
    Array, shape ``(N_C,)``, complex128.
    """
    k1, _ = jr.split(key)
    if N_C <= 6:
        if N_C > 1:
            R = max(min_sep / (2.0 * np.sin(np.pi / N_C)), scale * 0.5)
        else:
            R = 0.0
        coords = [R * np.exp(2j * np.pi * i / max(N_C, 1)) for i in range(N_C)]
    else:
        n_in = N_C // 2
        n_out = N_C - n_in
        R_in = max(min_sep / (2.0 * np.sin(np.pi / max(n_in, 2))), scale * 0.4)
        R_out = R_in + min_sep
        coords = (
            [R_in * np.exp(2j * np.pi * i / n_in) for i in range(n_in)]
            + [
                R_out * np.exp(2j * np.pi * i / n_out + 1j * np.pi / n_out)
                for i in range(n_out)
            ]
        )
    d = jnp.array(coords, dtype=jnp.complex128)
    return d + 0.1 * min_sep * complex_normal(k1, (N_C,)).astype(jnp.complex128)


def separation_penalty(
    d: Array,
    min_sep: float = 1.0,
    lam: float = 0.5,
) -> Array:
    r"""Soft repulsion :math:`\lambda \sum_{a<b} \max(0, s^2 - |d_a - d_b|^2)^2`.

    Penalises configurations whose minimum pairwise distance falls below
    ``min_sep``. Zero gradient outside the violation region.

    Parameters
    ----------
    d : Array, shape ``(A,)``
        Coherent-state positions.
    min_sep : float
        Target minimum pairwise distance.
    lam : float
        Penalty strength.

    Returns
    -------
    Array
        Scalar penalty.
    """
    A = d.shape[0]
    da = d.reshape(A, 1)
    db = d.reshape(1, A)
    viol = jnp.maximum(0.0, min_sep ** 2 - jnp.abs(da - db) ** 2)
    mask = 1.0 - jnp.eye(A)
    return lam * jnp.sum(mask * viol ** 2) / 2.0


def _make_scan_runner(
    loss_fn: Callable[[Any], Array],
    optimizer: optax.GradientTransformation,
    steps: int,
) -> Callable[[Any], tuple[Any, Array]]:
    """Compile a ``(params) -> (final_params, loss_curve)`` scan loop.

    Wirtinger conjugation and NaN sanitisation are applied to gradients
    on every step.
    """

    @jax.jit
    def run(params: Any) -> tuple[Any, Array]:
        opt_state = optimizer.init(params)

        def step(carry, _):
            params, opt_state = carry
            val, grads = jax.value_and_grad(loss_fn)(params)
            grads = jax.tree.map(jnp.conj, grads)
            grads = jax.tree.map(lambda g: jnp.nan_to_num(g, nan=0.0), grads)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), val

        (params, _), losses = jax.lax.scan(
            step, (params, opt_state), None, length=steps
        )
        return params, losses

    return run


def optimize_Fe_floating(
    gamma: float,
    N_C: int = 10,
    N_D: int = 10,
    D: int = 2,
    restarts: int = 40,
    steps_p1: int = 5000,
    steps_p2: int = 2000,
    lr_p1: float = 5e-3,
    lr_p2: float = 5e-4,
    sep_lam: float = 0.5,
    min_sep: float = 1.0,
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, Any]:
    r"""Optimize the floating-basis encoder + CPTP decoder to maximise :math:`F_e`.

    Two-phase Adam: Phase 1 explores with a soft separation penalty on
    the coherent positions to keep the Gram matrix well-conditioned;
    Phase 2 refines without the penalty at a smaller learning rate.
    Restarts whose final :math:`F_e` falls outside :math:`[0, 1]` (NaN
    or unphysical) are rejected.

    Parameters
    ----------
    gamma : float
        Pure-loss rate.
    N_C : int
        Number of coherent states in the encoding.
    N_D : int
        Kraus rank of the decoder.
    D : int
        Logical Hilbert-space dimension.
    restarts : int
    steps_p1, steps_p2 : int
    lr_p1, lr_p2 : float
    sep_lam : float
        Separation-penalty strength (Phase 1 only).
    min_sep : float
        Target minimum pairwise distance.
    seed : int
        PRNG seed.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys
        ``X`` ``(A, D)``, ``d`` ``(A,)``, ``Z`` ``(N_D, D, A)`` --
        best params (None if all restarts diverged).
        ``Fe`` -- best entanglement fidelity (float).
        ``nbar`` -- mean photon number of the maximally mixed encoded
        logical (float).
        ``curve`` -- concatenated Phase 1 + Phase 2 loss curve of the
        best restart (None if all diverged).
        ``n_valid`` -- number of accepted restarts.
    """
    opt1 = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_p1))
    opt2 = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_p2))

    def loss_p1(params):
        X, d, Z = params
        return (
            -entanglement_fidelity_pureloss(X, d, Z, gamma)
            + separation_penalty(d, min_sep, sep_lam)
        )

    def loss_p2(params):
        X, d, Z = params
        return -entanglement_fidelity_pureloss(X, d, Z, gamma)

    @jax.jit
    def eval_Fe(params):
        X, d, Z = params
        return entanglement_fidelity_pureloss(X, d, Z, gamma)

    run_p1 = _make_scan_runner(loss_p1, opt1, steps_p1)
    run_p2 = _make_scan_runner(loss_p2, opt2, steps_p2)

    # Compile-trigger pass on a dummy parameter set
    key0 = jr.PRNGKey(seed)
    k1, k2, k3 = jr.split(key0, 3)
    dummy = (
        (0.5 * complex_normal(k1, (N_C, D))).astype(jnp.complex128),
        init_separated_d(k2, N_C, min_sep),
        (0.5 * complex_normal(k3, (N_D, D, N_C))).astype(jnp.complex128),
    )
    _ = run_p1(dummy)
    _ = run_p2(dummy)
    if verbose:
        print(
            f"  JIT compiled (gamma={gamma:.3f}, N_C={N_C}, N_D={N_D}, "
            f"restarts={restarts}, steps={steps_p1}+{steps_p2})"
        )

    best_Fe, best_params, best_curve = -1.0, None, None
    n_valid = 0
    t0 = time.time()
    for r in range(restarts):
        key = jr.PRNGKey(seed * 1009 + r * 31 + 1)
        k1, k2, k3 = jr.split(key, 3)
        params0 = (
            (0.5 * complex_normal(k1, (N_C, D))).astype(jnp.complex128),
            init_separated_d(k2, N_C, min_sep),
            (0.5 * complex_normal(k3, (N_D, D, N_C))).astype(jnp.complex128),
        )
        params1, curve1 = run_p1(params0)
        Fe1 = float(eval_Fe(params1))
        if not (np.isfinite(Fe1) and 0.0 <= Fe1 <= 1.0):
            continue
        params2, curve2 = run_p2(params1)
        Fe2 = float(eval_Fe(params2))
        # If Phase 2 went unphysical or regressed, fall back to Phase 1.
        if not (np.isfinite(Fe2) and 0.0 <= Fe2 <= 1.0) or Fe2 < Fe1:
            Fe_final, params_final = Fe1, params1
            curve_final = np.asarray(curve1)
        else:
            Fe_final, params_final = Fe2, params2
            curve_final = np.concatenate(
                [np.asarray(curve1), np.asarray(curve2)]
            )
        n_valid += 1
        if Fe_final > best_Fe:
            best_Fe, best_params, best_curve = Fe_final, params_final, curve_final
        if verbose and (r + 1) % max(1, restarts // 5) == 0:
            print(
                f"    restart {r+1:3d}/{restarts}: best F_e = {best_Fe:.6f}, "
                f"valid = {n_valid}"
            )
    if verbose:
        print(
            f"  done in {time.time() - t0:5.1f}s -- best F_e = {best_Fe:.6f}, "
            f"valid = {n_valid}/{restarts}"
        )
    if best_params is None:
        return {"X": None, "d": None, "Z": None, "Fe": -1.0, "nbar": 0.0,
                "curve": None, "n_valid": 0}
    nbar = float(nbar_logical(best_params[0], best_params[1]))
    return {
        "X": best_params[0],
        "d": best_params[1],
        "Z": best_params[2],
        "Fe": best_Fe,
        "nbar": nbar,
        "curve": best_curve,
        "n_valid": n_valid,
    }


def optimize_Ic_floating(
    gamma: float,
    N_C: int = 10,
    D: int = 2,
    restarts: int = 40,
    steps_p1: int = 5000,
    steps_p2: int = 5000,
    lr_p1: float = 1e-2,
    lr_p2: float = 1e-3,
    sep_lam: float = 0.3,
    min_sep: float = 1.0,
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, Any]:
    r"""Optimize the floating-basis encoder to maximise :math:`I_c`.

    The decoder is irrelevant for :math:`I_c` (it depends only on the
    encoder and the channel), so only :math:`(X, d)` are optimised. Same
    two-phase Adam structure as :func:`optimize_Fe_floating`. Restarts
    whose final :math:`I_c` falls outside :math:`(-0.01, 1.01]` qubits
    are rejected.

    Parameters
    ----------
    Same as :func:`optimize_Fe_floating` (minus ``N_D``).

    Returns
    -------
    dict with keys
        ``X`` ``(A, D)``, ``d`` ``(A,)`` -- best params.
        ``Ic`` -- best coherent information in qubits.
        ``nbar`` -- mean photon number.
        ``curve`` -- best-restart loss curve (negative I_c in qubits;
        Phase 1 includes the separation penalty).
        ``n_valid`` -- number of accepted restarts.
    """
    opt1 = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_p1))
    opt2 = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_p2))

    def loss_p1(params):
        X, d = params
        return (
            -coherent_information_pureloss(X, d, gamma)
            + separation_penalty(d, min_sep, sep_lam)
        )

    def loss_p2(params):
        X, d = params
        return -coherent_information_pureloss(X, d, gamma)

    @jax.jit
    def eval_Ic(params):
        X, d = params
        return coherent_information_pureloss(X, d, gamma)

    run_p1 = _make_scan_runner(loss_p1, opt1, steps_p1)
    run_p2 = _make_scan_runner(loss_p2, opt2, steps_p2)

    key0 = jr.PRNGKey(seed)
    k1, k2 = jr.split(key0, 2)
    dummy = (
        (0.5 * complex_normal(k1, (N_C, D))).astype(jnp.complex128),
        init_separated_d(k2, N_C, min_sep),
    )
    _ = run_p1(dummy)
    _ = run_p2(dummy)
    if verbose:
        print(
            f"  JIT compiled (gamma={gamma:.3f}, N_C={N_C}, "
            f"restarts={restarts}, steps={steps_p1}+{steps_p2})"
        )

    best_Ic, best_params, best_curve = -np.inf, None, None
    n_valid = 0
    t0 = time.time()
    for r in range(restarts):
        key = jr.PRNGKey(seed * 1009 + r * 31 + 1)
        k1, k2 = jr.split(key, 2)
        params0 = (
            (0.5 * complex_normal(k1, (N_C, D))).astype(jnp.complex128),
            init_separated_d(k2, N_C, min_sep),
        )
        params1, curve1 = run_p1(params0)
        Ic1 = float(eval_Ic(params1))
        if not (np.isfinite(Ic1) and -0.01 < Ic1 <= 1.01):
            continue
        params2, curve2 = run_p2(params1)
        Ic2 = float(eval_Ic(params2))
        if not (np.isfinite(Ic2) and -0.01 < Ic2 <= 1.01):
            continue
        n_valid += 1
        if Ic2 > best_Ic:
            best_Ic = Ic2
            best_params = params2
            best_curve = np.concatenate(
                [np.asarray(curve1), np.asarray(curve2)]
            )
        if verbose and (r + 1) % max(1, restarts // 5) == 0:
            print(
                f"    restart {r+1:3d}/{restarts}: best I_c = {best_Ic:.6f} q, "
                f"valid = {n_valid}"
            )
    if verbose:
        print(
            f"  done in {time.time() - t0:5.1f}s -- best I_c = {best_Ic:.6f} q, "
            f"valid = {n_valid}/{restarts}"
        )
    if best_params is None:
        return {"X": None, "d": None, "Ic": -np.inf, "nbar": 0.0,
                "curve": None, "n_valid": 0}
    nbar = float(nbar_logical(best_params[0], best_params[1]))
    return {
        "X": best_params[0],
        "d": best_params[1],
        "Ic": best_Ic,
        "nbar": nbar,
        "curve": best_curve,
        "n_valid": n_valid,
    }
