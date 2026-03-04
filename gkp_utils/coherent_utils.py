import jax.numpy as jnp
import jax.random as jr
import jax
import equinox as eqx
from jaxtyping import Array
from typing import Callable
from functools import partial
import sys
from typing import Any

sys.path.append("../..")
from gkp_utils.jax_analytic_utils import *
from gkp_utils.utils import dqdisplace, dqtrace, dqcoherent, GKP_N
import matplotlib.pyplot as plt


def charfunc_to_fock_basis(
    fn: Callable, x=jnp.linspace(-5, 5, 40), y=jnp.linspace(-5, 5, 40), N=GKP_N
):
    X, Y = jnp.meshgrid(x, y)
    Z = X.ravel() + 1j * Y.ravel()
    all_coeffs = jax.vmap(fn)(Z)
    all_displacements = jax.vmap(partial(dqdisplace, N))(Z)
    fock_map = jnp.einsum("a,aij->ij", all_coeffs, all_displacements)
    fock_map = fock_map / dqtrace(fock_map)
    return fock_map


def _plot_char_func(
    fn: Array,
    map_fn: Callable,
    x: Array,
    y: Array,
):
    X, Y = jnp.meshgrid(x, y)
    U = X.ravel() + 1j * Y.ravel()
    arr = jax.vmap(fn)(U)
    img = plt.imshow(map_fn(arr).reshape(X.shape), extent=[x[0], x[-1], y[0], y[-1]])
    plt.xticks([x[0], 0, x[-1]])
    plt.yticks([y[0], 0, y[-1]])
    plt.colorbar(img)
    plt.show()


def plot_char_func_real(
    fn: Array,
    x: Array = jnp.linspace(-4, 4, 30),
    y: Array = jnp.linspace(-4, 4, 30),
):
    _plot_char_func(fn, jnp.real, x, y)


def plot_char_func_imag(
    fn: Array,
    x: Array = jnp.linspace(-4, 4, 30),
    y: Array = jnp.linspace(-4, 4, 30),
):
    _plot_char_func(fn, jnp.imag, x, y)


class CoherentKet(eqx.Module):
    ds: Array
    cs: Array

    def __init__(self, cs, ds, renormalize=False):
        ca = cs.reshape(-1, 1)
        cb = cs.reshape(1, -1)
        da = ds.reshape(-1, 1)
        db = ds.reshape(1, -1)
        phi = jnp.real(db) * jnp.imag(da) - jnp.imag(db) * jnp.real(da)
        phase = jnp.exp(1j * phi)
        prefactor = ca * jnp.conj(cb)
        envelope = jnp.exp(-0.5 * (jnp.abs(da - db) ** 2))
        sqrtoverlap = jnp.sqrt(jnp.real(jnp.sum(phase * prefactor * envelope)))
        self.ds = ds
        self.cs = cs / sqrtoverlap

    @jax.jit
    def __call__(self, u: complex):
        N = self.cs.shape[0]
        ca = self.cs.reshape(1, N)
        da = self.ds.reshape(1, N)
        cb = self.cs.reshape(N, 1)
        db = self.ds.reshape(N, 1)
        phi_dabu = jnp.real(u) * jnp.imag(da + db) - jnp.imag(u) * jnp.real(da + db)
        phi_dab = jnp.real(da) * jnp.imag(db) - jnp.imag(da) * jnp.real(db)
        phase = jnp.exp(1j * (phi_dab + phi_dabu))
        envelope = jnp.exp(-0.5 * jnp.abs(db - da - u) ** 2)
        mat = jnp.conj(ca) * cb * phase * envelope
        return mat.sum()

    def to_fock_basis(self, N=GKP_N):
        coherents = jax.vmap(lambda alpha: dqcoherent(N, alpha))(self.ds)
        psi_weighted = jnp.einsum("ija,i->ja", coherents, self.cs).squeeze()
        return jnp.einsum("i,j->ij", psi_weighted, jnp.conj(psi_weighted))


def walsh(n):
    # int -> (n, 2^n)
    return ((jnp.arange(2**n)[:, None] >> jnp.arange(n - 1, -1, -1)) & 1).T


def walsh_2(n):
    return walsh(n) - 0.5


@jax.jit
def phase_from_displacement_sequence(d: Array):
    # d: (n,)
    # e^{i \phi}D(\sum_i d_i) = D(d_n)...D(d_1)D(d_0)
    n = d.shape[0]
    A = jnp.triu(jnp.ones((n, n))) - jnp.eye(n)
    a = jnp.stack((jnp.real(d), jnp.imag(d)), axis=1)
    Omega = jnp.array([[0.0, 1.0], [-1.0, 0.0]])
    phi = jnp.einsum("ij,ik,kl,jl->", A, a, Omega, a)
    return jnp.exp(1.0j * phi)


@jax.jit
def ECD_single_2x2(rots: Array):
    # (phi, theta): (2,)
    theta, phi = rots[1], rots[0]
    return jnp.array(
        [
            [
                jnp.sin(theta / 2) * (jnp.sin(phi) - 1.0j * jnp.cos(phi)),
                jnp.cos(theta / 2),
            ],
            [
                jnp.cos(theta / 2),
                jnp.sin(theta / 2) * (-jnp.sin(phi) - 1.0j * jnp.cos(phi)),
            ],
        ]
    )


batch_ECD_single_2x2 = jax.jit(jax.vmap(ECD_single_2x2))


@jax.jit
def ECD_2x2(rots: Array):
    # (phi, theta): (n,2)
    submatrices = batch_ECD_single_2x2(rots)
    R_tot = jnp.eye(2, dtype=jnp.complex64)

    @jax.jit
    def body(i, R):
        return R @ submatrices[i]

    return jax.lax.fori_loop(0, rots.shape[0], body, R_tot)


def params_to_charfunc(circuit_parameters, N_l):
    alpha_beta = g(circuit_parameters, N_l)  # TODO
    # print(alpha_beta)
    alpha = alpha_beta[0]
    beta_j = alpha_beta[1, 0].reshape(-1, 1)  # (N,1)
    beta_k = beta_j.T  # (1,N)

    @jax.jit
    def char_func_i(i: int, u: complex):
        envelope = jnp.exp(-0.5 * jnp.abs(beta_k + u - beta_j) ** 2)
        phiu_jk = jnp.real(u) * jnp.imag(beta_k + beta_j) - jnp.imag(u) * jnp.real(
            beta_k + beta_j
        )
        phi_jk = jnp.real(beta_k) * jnp.imag(beta_j) - jnp.imag(beta_k) * jnp.real(
            beta_j
        )
        phase = jnp.exp(1j * (phiu_jk + phi_jk))
        alpha_j = alpha[i].reshape(-1, 1)
        alpha_k = alpha_j.T
        alpha_jk = alpha_j * jnp.conj(alpha_k)
        return (alpha_jk * envelope * phase).sum()

    @jax.jit
    def char_func_tot(u: complex):
        return char_func_i(0, u) + char_func_i(1, u)

    return char_func_tot


@jax.jit
def analytic_fidelity_i_2(
    coeffs_a: Array, coeffs_b: Array, peaks_a: Array, peaks_b: Array
):
    A = peaks_a.shape[0]
    B = peaks_b.shape[0]
    ca = coeffs_a.reshape(A, 1)
    da = peaks_a.reshape(A, 1)
    alphaj = coeffs_b.reshape(1, B)
    betaj = peaks_b.reshape(1, B)

    prefactor = jnp.conj(ca) * alphaj
    envelope = jnp.exp(-0.5 * jnp.abs(betaj - da) ** 2)
    phi = jnp.real(da) * jnp.imag(betaj) - jnp.imag(da) * jnp.real(betaj)
    phase = jnp.exp(1j * phi)
    return jnp.abs(jnp.sum(prefactor * envelope * phase)) ** 2


@jax.jit
def analytic_fidelity_2(
    all_coeffs_a: Array, all_coeffs_b: Array, all_peaks_a: Array, all_peaks_b: Array
):
    N = all_peaks_a.shape[0]
    M = all_peaks_b.shape[0]

    total = 0.0
    for i in range(N):
        for j in range(M):
            total += analytic_fidelity_i_2(
                coeffs_a=all_coeffs_a[i],
                coeffs_b=all_coeffs_b[j],
                peaks_a=all_peaks_a[i],
                peaks_b=all_peaks_b[j],
            )
    return total


@partial(jax.jit, static_argnums=2)
def analytic_fidelity_2_wrapper(
    coherent: CoherentKet, circuit_parameters: Array, N_l: int
):
    alpha_coherent = jnp.expand_dims(coherent.cs, 0)
    beta_coherent = jnp.expand_dims(coherent.ds, 0)

    alpha_beta_circuit = g(circuit_parameters, N_l)
    alpha_circuit = alpha_beta_circuit[0]
    beta_circuit = alpha_beta_circuit[1]

    return analytic_fidelity_2(
        all_coeffs_a=alpha_coherent,
        all_coeffs_b=alpha_circuit,
        all_peaks_a=beta_coherent,
        all_peaks_b=beta_circuit,
    )


def optimize_wrt_state(target_state: CoherentKet, lr=0.005, steps=10000, restarts=5):
    @partial(jax.jit, static_argnums=0)
    def analytic_loss_fn(N_l: int, circuit_params: Array):
        return 1.0 - analytic_fidelity_2_wrapper(target_state, circuit_params, N_l)

    @partial(jax.jit, static_argnums=2)
    def analytic_train_step(a: Array, opt_state: Any, N_l: int):
        a = a.astype(jnp.complex64)
        partial_analytic_loss_fn = partial(analytic_loss_fn, N_l)
        grads = jax.grad(partial_analytic_loss_fn)(a)
        grads = jnp.conj(grads)  # AAHHHHH
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
        a_init = jnp.zeros((N_depth, 3), jnp.complex64)
        a_init = a_init.at[:, 1:].set(
            2 * jnp.pi * jr.uniform(key=k2, shape=(N_depth, 2))
        )
        a_init = a_init.at[:, 0].set(
            4 * jr.normal(key=k1, shape=(N_depth,))
            + 4.0j * jr.normal(key=k3, shape=(N_depth,))
        )
        a = a_init
        # print(a)
        opt_state = optimizer.init(a_init)
        for step_i in range(steps):
            a, opt_state = analytic_train_step(a=a, opt_state=opt_state, N_l=N_layer)
            if step_i % 100 == 0:
                current_loss = analytic_loss_fn(N_layer, a)
                loss_vals = loss_vals.at[step_i].set(current_loss.item())
                print(
                    f"Restart {restart}, Step {step_i}, Loss = {current_loss.item():.6f}"
                )
        current_loss = analytic_loss_fn(N_l=N_layer, circuit_params=a)
        if current_loss < best_loss:
            best_loss = current_loss
            best_val = a
            print(restart, "new best", best_loss)
            print(best_val)

    return best_val, best_loss
