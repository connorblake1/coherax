"""Gradient-based optimization tools that leverage the coherent-basis native tools."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import jax
import optax
from jaxtyping import Array
from typing import Any
from functools import partial

from coherax.states import CoherentKet
from coherax.fidelity import (
    analytic_fidelity_wrapper,
    analytic_fidelity_transfer_wrapper,
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
        a_init = (
            a_init.at[:, 1:3].add(
                2 * random_angle * jr.uniform(key=k2, shape=(N_depth, 2))
            )
            - random_angle
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
