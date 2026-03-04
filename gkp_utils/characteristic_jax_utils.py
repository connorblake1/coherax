import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg as jla
import dynamiqs as dq
from functools import partial
import equinox as eqx
from jaxtyping import Array
from typing import Callable, Any
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import optax
import math
# TODO: replace all partial(jax.jit,...) with filterjit
# TODO: give each equation an eq number from the final paper


@jax.jit
def dqtensor(*args):
    return dq.tensor(*args).to_jax()


@jax.jit
def dqdag(arg):
    return dq.dag(arg).to_jax()


@jax.jit
def dag(arr):
    return jnp.conj(arr.T)


def dqeye(arg):
    return dq.eye(arg).to_jax()


def dqnumber(arg):
    return dq.number(arg).to_jax()


def dqdestroy(arg):
    return dq.destroy(arg).to_jax()


def dqcreate(arg):
    return dq.create(arg).to_jax()


@jax.jit
def dqtrace(arg):
    return dq.trace(arg)


@partial(jax.jit, static_argnums=0)
def dqdisplace(*args):
    return dq.displace(*args).to_jax()


@partial(jax.jit, static_argnums=0)
def dqsqueeze(*args):
    return dq.squeeze(*args).to_jax()


@partial(jax.jit, static_argnums=0)
def dqfock_dm(*args):
    return dq.fock_dm(*args).to_jax()


@partial(jax.jit, static_argnums=0)
def dqcoherent_dm(*args):
    return dq.coherent_dm(*args).to_jax()


@partial(jax.jit, static_argnums=0)
def dqcoherent(*args):
    return dq.coherent(*args).to_jax()


@partial(jax.jit, static_argnums=(1, 2))
def dqptrace(*args):
    return dq.ptrace(*args).to_jax()


@jax.jit
def dqexpect(*args):
    return dq.expect(*args)


@jax.jit
def dqtodm(*args):
    return dq.todm(*args).to_jax()


@jax.jit
def aOmegab(a: Array, b: Array):
    return jnp.real(a) * jnp.imag(b) - jnp.imag(a) * jnp.real(b)


@jax.jit
def e_n1iaOmegab(a: Array, b: Array):
    return jnp.exp(-1j * aOmegab(a, b))


GKP_N = 100
root2 = jnp.sqrt(2.0)
IN = dqeye(GKP_N)
I2 = dqeye(2)
sigma_x = sigma_x = dq.sigmax().to_jax()
sigma_y = dq.sigmay().to_jax()
sigma_z = dq.sigmaz().to_jax()
n_hat = dqnumber(GKP_N)
a = dqdestroy(GKP_N)
a_dag = dqcreate(GKP_N)
x = (a + a_dag) / root2
p = -1.0j * (a - a_dag) / root2
ket0 = dq.fock(2, 0)
ket1 = dq.fock(2, 1)


@jax.jit
def W(u: complex):
    return dqdisplace(GKP_N, u * jnp.sqrt(jnp.pi))


@jax.jit
def CD(u: complex):
    return jla.expm(dqtensor(u * a_dag - jnp.conj(u) * a, sigma_z))


@jax.jit
def ECD(beta: complex):
    return dqtensor(IN, sigma_x) @ CD(beta / 2)


def R_x(theta: float):
    return jla.expm(-1j * theta * sigma_x / 2)


def R_y(theta: float):
    return jla.expm(-1j * theta * sigma_y / 2)


def R_z(theta: float):
    return jla.expm(-1j * theta * sigma_z / 2)


@jax.jit
def qubit_rotation(
    phi: float,
    theta: float,
    gamma: float,
):
    return jnp.exp(-0.5j * (gamma + phi)) * jnp.array(
        [
            [jnp.cos(theta / 2), -jnp.sin(theta / 2) * jnp.exp(1.0j * gamma)],
            [
                jnp.sin(theta / 2) * jnp.exp(1.0j * phi),
                jnp.cos(theta / 2) * jnp.exp(1.0j * (phi + gamma)),
            ],
        ]
    )


@jax.jit
def ecd_rotation_2x2(
    phi: float,
    theta: float,
    gamma: float,
):
    return jnp.exp(-0.5j * (gamma + phi)) * jnp.array(
        [
            [
                jnp.sin(theta / 2) * jnp.exp(1.0j * phi),
                jnp.cos(theta / 2) * jnp.exp(1.0j * (phi + gamma)),
            ],
            [jnp.cos(theta / 2), -jnp.sin(theta / 2) * jnp.exp(1.0j * gamma)],
        ]
    )


@partial(jax.jit, static_argnums=1)
def circuit_layer(layer: Array, N=GKP_N):
    d = layer[0]
    phi = jnp.real(layer[1])
    theta = jnp.real(layer[2])
    gamma = jnp.real(layer[3])
    return ECD(beta=d) @ dqtensor(
        jnp.eye(N), qubit_rotation(phi=phi, theta=theta, gamma=gamma)
    )


@jax.jit
def compose_ECD_layers(params: Array):
    circuit = circuit_layer(params[0, :])

    def body_mult(i, circ):
        return circuit_layer(params[i, :]) @ circ

    return jax.lax.fori_loop(1, params.shape[0], body_mult, circuit)


@partial(jax.jit, static_argnums=1)
def traceout_unitary(U: Array, N=GKP_N):
    K = jnp.zeros((2, N, N), jnp.complex64)
    K = K.at[0, :, :].set(
        dqtensor(jnp.eye(N), dqdag(dq.fock(2, 0)))
        @ U
        @ dqtensor(jnp.eye(N), dq.fock(2, 0))
    )
    K = K.at[1, :, :].set(
        dqtensor(jnp.eye(N), dqdag(dq.fock(2, 1)))
        @ U
        @ dqtensor(jnp.eye(N), dq.fock(2, 0))
    )
    return K


@jax.jit
def circuit_params_to_2channel(params: Array):
    return traceout_unitary(compose_ECD_layers(params))


batch_circuit_params_to_2channel = jax.jit(jax.vmap(circuit_params_to_2channel))


@partial(jax.jit, static_argnums=(1,))
def indices_for_bit(k: int, K: int, bit: int) -> Array:
    """
    This is written in a very opaque way because it needs to be jax-traceable.
    Its function is to return the indices of where a binary Hadamard matrix is equal to 0.
    This then indexes into the Kraus map tree to build the individual operators of the supermap.
    """
    n = 1 << K
    b = 1 << (K - k - 1)
    q = jnp.arange(n >> 1)
    base = q + (q // b) * b
    return base + bit * b


@partial(jax.jit, static_argnums=1)
def compact_channel_to_exec_channel(operators: Array, K: int):
    # operators: (K,2,GKP_N,GKP_N) ie the output of batch_circuit_params_to_2channel
    total_channel = jnp.broadcast_to(jnp.eye(GKP_N), (2**K, GKP_N, GKP_N)).astype(
        jnp.complex64
    )  # (2^K, GKP_N, GKP_N) ie the apply_kraus_map channel
    for i in range(K):  # gets unrolled in jit
        bp = indices_for_bit(i, K, 0)
        cbp = indices_for_bit(i, K, 1)
        total_channel = total_channel.at[bp, :, :].set(
            operators[i, 0, :, :] @ total_channel[bp, :, :]
        )
        total_channel = total_channel.at[cbp, :, :].set(
            operators[i, 1, :, :] @ total_channel[cbp, :, :]
        )
    return total_channel


@jax.jit
def apply_kraus_map_nonorm(ops: Array, rho: Array):
    # ops: (K, N, N)
    # rho: (N, N)
    return jnp.sum(jax.jit(jax.vmap(lambda op: op @ rho @ dqdag(op)))(ops), axis=0)


@jax.jit
def apply_kraus_map(ops: Array, rho: Array):
    # ops: (K, N, N)
    # rho: (N, N)
    rho_out = jnp.sum(jax.jit(jax.vmap(lambda op: op @ rho @ dqdag(op)))(ops), axis=0)
    return rho_out / dqtrace(rho_out)


@jax.jit
def apply_kraus_map_n(ops: Array, rho: Array, n: int):
    # ops: (K, N, N)
    # rho: (N, N)
    def body_loop(i, rho_loop):
        return apply_kraus_map(ops, rho_loop)

    rho_out = jax.lax.fori_loop(0, n, body_loop, rho)
    return rho_out / dqtrace(rho_out)


@jax.jit
def compose_channel_kraus(ch1, ch2):
    # ch1,ch2: (K, N, N)
    if ch1.shape[2] != ch2.shape[1]:
        return None
    new_size = ch1.shape[0] * ch2.shape[0]
    new_ops = jnp.zeros((new_size, ch1.shape[1], ch2.shape[2]), dtype="complex64")
    for i in range(ch1.shape[0]):
        for j in range(ch2.shape[0]):
            new_ops = new_ops.at[i * ch2.shape[0] + j, :, :].set(
                ch1[i, :, :] @ ch2[j, :, :]
            )
    return new_ops


@partial(jax.jit, static_argnums=1)
def addmask(n: int, N_l: int):
    index = 2**n
    mask = jnp.arange(N_l) < index
    return mask.astype(jnp.complex64)


@partial(jax.jit, static_argnums=1)
def caddmask(n: int, N_l: int):
    return jnp.roll(addmask(n, N_l), 2**n, axis=0)


class TraceoutLayer(eqx.Module):
    pass


class TraceoutLayer(eqx.Module):
    alphas: Array
    betas: Array
    n: int  # this is how many layers it holds
    N_l: int

    def __init__(
        self, n: int, N_l: int, alphas: Array | None = None, betas: Array | None = None
    ):
        self.n = n
        self.N_l = N_l
        if alphas is None:
            self.alphas = jnp.zeros((2, 2, N_l), jnp.complex64)
        else:
            self.alphas = alphas
        if betas is None:
            self.betas = jnp.zeros((2, 2, N_l), jnp.complex64)
        else:
            self.betas = betas

    @staticmethod
    @partial(jax.jit, static_argnums=1)
    def from_single_param(circuit_layer: Array, N_l: int) -> TraceoutLayer:
        d = circuit_layer[0].astype(jnp.complex64)
        phi = jnp.real(circuit_layer[1])
        theta = jnp.real(circuit_layer[2])
        gamma = jnp.real(circuit_layer[3])
        alphas = jnp.zeros((2, 2, N_l), dtype=jnp.complex64)
        betas = jnp.zeros((2, 2, N_l), dtype=jnp.complex64)

        alphas = alphas.at[:, :, 0].set(
            ecd_rotation_2x2(phi=phi, theta=theta, gamma=gamma)
        )
        betas = betas.at[:, :, 0].set(jnp.array([[-d / 2, -d / 2], [d / 2, d / 2]]))
        return TraceoutLayer(n=1, N_l=N_l, alphas=alphas, betas=betas)

    @staticmethod
    @partial(jax.jit, static_argnums=2)
    def unitarycompose(l_a: TraceoutLayer, l_b: TraceoutLayer, N_l: int):
        # assumes l_a pure Layer # (n_a = 1)
        n_b: int = l_b.n - 1
        n_add = addmask(n_b, N_l)
        n_cadd = caddmask(n_b, N_l)
        full_add = addmask(n_b + 1, N_l)
        cshift = lambda x: jnp.roll(x, 2**n_b, axis=0)

        d = l_b.betas
        c = l_b.alphas
        alphas = l_a.alphas
        betas = l_a.betas

        d2_minus = betas[0, 0, 0] * full_add
        d2_plus = betas[1, 1, 0] * full_add

        dcol0 = n_add * d[0, 0, :] + n_cadd * cshift(d[1, 0, :])
        dcol1 = n_add * d[0, 1, :] + n_cadd * cshift(d[1, 1, :])

        out_betas = jnp.array(
            [[d2_minus + dcol0, d2_minus + dcol1], [d2_plus + dcol0, d2_plus + dcol1]]
        )

        c00 = (
            alphas[0, 0, 0]
            * n_add
            * c[0, 0, :]
            * e_n1iaOmegab(betas[0, 0, 0] * full_add, n_add * d[0, 0, :])
        ) + cshift(
            alphas[0, 1, 0]
            * n_add
            * c[1, 0, :]
            * e_n1iaOmegab(betas[0, 1, 0] * n_add, d[1, 0, :])
        )

        c01 = (
            alphas[0, 0, 0]
            * n_add
            * c[0, 1, :]
            * e_n1iaOmegab(betas[0, 0, 0] * full_add, n_add * d[0, 1, :])
        ) + cshift(
            alphas[0, 1, 0]
            * n_add
            * c[1, 1, :]
            * e_n1iaOmegab(betas[0, 1, 0] * n_add, d[1, 1, :])
        )

        c10 = (
            alphas[1, 0, 0]
            * n_add
            * c[0, 0, :]
            * e_n1iaOmegab(betas[1, 0, 0] * full_add, n_add * d[0, 0, :])
        ) + cshift(
            alphas[1, 1, 0]
            * n_add
            * c[1, 0, :]
            * e_n1iaOmegab(betas[1, 1, 0] * n_add, d[1, 0, :])
        )

        c11 = (
            alphas[1, 0, 0]
            * n_add
            * c[0, 1, :]
            * e_n1iaOmegab(betas[1, 0, 0] * full_add, n_add * d[0, 1, :])
        ) + cshift(
            alphas[1, 1, 0]
            * n_add
            * c[1, 1, :]
            * e_n1iaOmegab(betas[1, 1, 0] * n_add, d[1, 1, :])
        )

        out_alphas = jnp.array([[c00, c01], [c10, c11]], dtype=jnp.complex64)

        return TraceoutLayer(
            n=l_b.n + 1, N_l=alphas.shape[2], alphas=out_alphas, betas=out_betas
        )

    @staticmethod
    @jax.jit
    def to_traceout(l: TraceoutLayer) -> Array:
        return l.alphas[:, 0, :], l.betas[:, 0, :]

    @staticmethod
    @partial(jax.jit, static_argnums=1)
    def from_params(circuit_params: Array, N_l: int) -> TraceoutLayer:
        circuit = TraceoutLayer.from_single_param(
            circuit_layer=circuit_params[0, :], N_l=N_l
        )

        @jax.jit
        def body_compose(i: int, circuit):
            return TraceoutLayer.unitarycompose(
                l_a=TraceoutLayer.from_single_param(
                    circuit_layer=circuit_params[i, :], N_l=N_l
                ),
                l_b=circuit,
                N_l=N_l,
            )

        return jax.lax.fori_loop(
            1, circuit_params.shape[0], body_fun=body_compose, init_val=circuit
        )


@partial(jax.jit, static_argnums=1)
def g(circuit_params: Array, N_l: int):
    return TraceoutLayer.to_traceout(TraceoutLayer.from_params(circuit_params, N_l))


def channel_from_b(alphas: Array, betas: Array):
    ops = jnp.zeros((alphas.shape[0], GKP_N, GKP_N), dtype=jnp.complex64)

    @jax.jit
    def sum_displacements_over_i(j):
        @jax.jit
        def body_fun(i, partial_sum):
            return partial_sum + alphas[j, i] * dqdisplace(GKP_N, betas[j, i]).astype(
                jnp.complex64
            )

        return jax.lax.fori_loop(
            0, alphas.shape[1], body_fun, jnp.zeros((GKP_N, GKP_N), dtype=jnp.complex64)
        )

    @jax.jit
    def outer_body_fun(j, ops_accum):
        return ops_accum.at[j, :, :].set(sum_displacements_over_i(j))

    ops = jax.lax.fori_loop(0, alphas.shape[0], outer_body_fun, ops)
    return ops


def krauscompose(l_a: TraceoutLayer, l_b: TraceoutLayer):
    # TODO deprecate
    alphas_a, betas_a = TraceoutLayer.to_traceout(l_a)
    alphas_b, betas_b = TraceoutLayer.to_traceout(l_b)
    t_A = alphas_a.shape[0]
    t_B = alphas_b.shape[0]
    t_AB = t_A * t_B
    N_A = alphas_a.shape[1]
    N_B = alphas_b.shape[1]
    N_AB = N_A * N_B
    betas_a_expanded = jnp.broadcast_to(betas_a[:, None, :, None], (t_A, t_B, N_A, N_B))
    betas_b_expanded = jnp.broadcast_to(betas_b[None, :, None, :], (t_A, t_B, N_A, N_B))
    new_alphas = jnp.einsum(
        "aj,bk,abjk->abjk",
        alphas_a,
        alphas_b,
        e_n1iaOmegab(betas_a_expanded, betas_b_expanded),
    )
    new_betas = betas_a_expanded + betas_b_expanded
    return new_alphas.reshape((t_AB, N_AB)), new_betas.reshape((t_AB, N_AB))


@partial(jax.jit, static_argnums=(1, 2))
def super_g(super_circuit_params: Array, N_l: int, T: int):
    # super_circuit_params (T,n,4)
    # assume N_l == 2**n
    alphas_precompose = jnp.zeros((T, 2, N_l), jnp.complex64)
    betas_precompose = jnp.zeros((T, 2, N_l), jnp.complex64)
    for i in range(T):  # gets unrolled
        a, b = g(circuit_params=super_circuit_params[i], N_l=N_l)
        alphas_precompose = alphas_precompose.at[i, :, :].set(a)
        betas_precompose = betas_precompose.at[i, :, :].set(b)
    alpha_total = jnp.zeros((2**T, N_l**T), jnp.complex64)
    beta_total = jnp.zeros((2**T, N_l**T), jnp.complex64)

    N_filled = N_l
    alpha_total = alpha_total.at[:2, :N_filled].set(alphas_precompose[0])
    beta_total = beta_total.at[:2, :N_filled].set(betas_precompose[0])
    for i in range(1, T):
        alpha_total = alpha_total.at[: 2 ** (i + 1), : N_filled * N_l].set(
            (
                alphas_precompose[i, :, None, :, None]
                * alpha_total[None, : (2**i), None, :N_filled]
                * e_n1iaOmegab(
                    betas_precompose[i, :, None, :, None],
                    beta_total[None, : (2**i), None, :N_filled],
                )
            ).reshape((2 ** (i + 1), N_filled * N_l))
        )
        beta_total = beta_total.at[: 2 ** (i + 1), : N_filled * N_l].set(
            (
                betas_precompose[i, :, None, :, None]
                + beta_total[None, : (2**i), None, :N_filled]
            ).reshape((2 ** (i + 1), N_filled * N_l))
        )
        N_filled = N_filled * N_l
    return alpha_total, beta_total


## COHERENT UTILITIES
def charfunc_to_fock_basis(
    fn: Callable,
    x=jnp.linspace(-5, 5, 40),
    y=jnp.linspace(-5, 5, 40),
    N=GKP_N,
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
    fig, ax = plt.subplots()
    img = ax.imshow(
        map_fn(arr).reshape(X.shape),
        extent=(x[0], x[-1], y[0], y[-1]),
        origin="lower",
        aspect="auto",
        cmap="bwr",
    )
    # ax.set_xticks([x[0], 0, x[-1]])
    # ax.set_yticks([y[0], 0, y[-1]])
    cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.04)
    cbar = fig.colorbar(img, cax=cax)
    return fig, ax, cbar


def plot_char_func_real(
    fn: Array,
    x: Array = jnp.linspace(-4, 4, 30),
    y: Array = jnp.linspace(-4, 4, 30),
):
    return _plot_char_func(fn, jnp.real, x, y)


def plot_char_func_imag(
    fn: Array,
    x: Array = jnp.linspace(-4, 4, 30),
    y: Array = jnp.linspace(-4, 4, 30),
):
    return _plot_char_func(fn, jnp.imag, x, y)


@jax.jit
def coherent_overlap(alpha: Array, beta: Array):
    # <alpha|beta>
    # alpha,beta should be broadcastable
    return jnp.exp(-0.5 * jnp.abs(alpha - beta) ** 2 + 1.0j * aOmegab(alpha, beta))


class CoherentKet(eqx.Module):
    ds: Array
    cs: Array

    def __init__(self, cs, ds):
        # TODO rewrite cleanly
        ca = cs.reshape(-1, 1)
        cb = cs.reshape(1, -1)
        da = ds.reshape(-1, 1)
        db = ds.reshape(1, -1)
        phase = jnp.exp(1j * aOmegab(db, da))
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
        envelope = jnp.exp(
            -0.5 * jnp.abs(db - da - u) ** 2
            + 1j * (aOmegab(da, db) + aOmegab(u, da + db))
        )
        return jnp.sum(jnp.conj(ca) * cb * envelope)

    @jax.jit  # TODO works?
    def to_fock_basis(self, N: int = GKP_N):
        coherents = jax.vmap(lambda alpha: dqcoherent(N, alpha))(self.ds)
        psi_weighted = jnp.einsum(
            "ija,i->ja", coherents, self.cs
        ).squeeze()  # TODO works?
        return jnp.einsum("i,j->ij", psi_weighted, jnp.conj(psi_weighted))


class CoherentDM(eqx.Module):
    ds: Array
    C: Array

    def __init__(self, C, ds):
        # TODO test
        G = coherent_overlap(ds.reshape((-1, 1)), ds.reshape((1, -1)))
        C = C / jnp.einsum("ij,ji", C, G)
        self.C = C
        self.ds = ds

    @partial(jax.jit, static_argnums=1)
    def to_fock_basis(self, N: int = GKP_N):
        coherents = jnp.squeeze(
            jax.vmap(lambda alpha: dqcoherent(N, alpha))(self.ds)
        )  # (A,N)
        return jnp.einsum("ab,ai,bj->ij", self.C, coherents, jnp.conj(coherents))

    @staticmethod
    def from_ket(state: CoherentKet):
        return CoherentDM(
            C=jnp.einsum("i,j->ij", state.cs, jnp.conj(state.cs)), ds=state.ds
        )


@jax.jit
def invsqrtm(A: Array):
    w, v = jnp.linalg.eigh(A)
    return (v / jnp.sqrt(w)) @ dag(v)


def sparse_eigh(O: Array, eps: float = 1e-6):
    lambda_O, U_O = jnp.linalg.eigh(O)
    mask = lambda_O >= eps
    return lambda_O[mask], U_O[:, mask]


def sparse_tensor_eigh(T, eps: float = 1e-6):
    # T: (A,A,A,A) where it's block Hermitian: T_{aba'b'} = T_{a'b'ab}^*
    A = T.shape[0]
    M = jnp.reshape(T, (A * A, A * A))
    w, U = sparse_eigh(M, eps=eps)
    chis = jnp.reshape(U, (A, A, w.shape[0]))
    return w, chis


class BosonicSubspace(eqx.Module):
    ds: Array
    G: Array
    lambda_G: Array
    U_G: Array
    T: Array
    Tp: Array

    def __init__(self, ds, eps=1e-6):
        A = ds.shape[0]
        G = coherent_overlap(ds.reshape((A, 1)), ds.reshape((1, A)))
        lambda_G, U_G = sparse_eigh(G, eps)
        self.T = U_G @ jnp.diag(lambda_G**-0.5)
        self.Tp = jnp.diag(lambda_G**0.5) @ dag(U_G)

        self.ds = ds
        self.G = G
        self.lambda_G = lambda_G
        self.U_G = U_G

    def op_c2o_transform(self, O: Array):
        # O: (A,A) in coherent basis -> orthogonal basis
        # A = \sum a_{ij} \ket{\d_i}\bra{\d_j} to
        # A = \sum a'_{ij} \ket{\phi_i}\bra{\phi_j}
        return jnp.einsum("ia,ab,jb->ij", self.Tp, O, jnp.conj(self.Tp))

    def op_o2c_transform(self, O: Array):
        # O: (A,A) in orthogonal basis -> coherent basis
        return jnp.einsum("ai,ij,bj->ab", self.T, O, jnp.conj(self.T))

    def ket_c2o_transform(self, ket: Array):
        # ket: (A,) in coherent basis -> ortho
        return jnp.einsum("ia,a->i", self.Tp, ket)

    def ket_o2c_transform(self, ket: Array):
        # ket: (A,) in ortho basis -> coherent
        return jnp.einsum("ai,i->a", self.T, ket)

    def synthesize_ket_fock(self, ket, N: int = GKP_N):
        coeffs = jnp.squeeze(
            jax.vmap(lambda alpha: dqcoherent(N, alpha))(self.ds)
        )  # (A,N)
        return jnp.einsum("ai,a->i", coeffs, ket)

    @partial(jax.jit, static_argnums=2)
    def op_to_fock(self, O: Array, N: int = GKP_N):
        # O: (A,A) in coherent basis
        coherents = jnp.squeeze(
            jax.vmap(lambda alpha: dqcoherent(N, alpha))(self.ds)
        )  # (A,N)
        return jnp.einsum("ai,bj,ab->ij", coherents, jnp.conj(coherents), O)


@jax.jit
def phase_from_displacement_sequence(d: Array):
    # d: (n,)
    # e^{i \phi}D(\sum_i d_i) = D(d_n)...D(d_1)
    n = d.shape[0]
    A = jnp.triu(jnp.ones((n, n))) - jnp.eye(n)
    a = jnp.stack((jnp.real(d), jnp.imag(d)), axis=1)
    Omega = jnp.array([[0.0, 1.0], [-1.0, 0.0]])
    phi = jnp.einsum("ij,ik,kl,jl->", A, a, Omega, a)
    return jnp.exp(1.0j * phi)


def params_to_charfunc(circuit_parameters: Array, N_l: int):
    if len(circuit_parameters.shape) == 2:
        alpha, beta = g(circuit_parameters, N_l)
    else:
        alpha, beta = super_g(
            circuit_parameters, N_l=N_l, T=circuit_parameters.shape[0]
        )

    @jax.jit
    def char_func_i(i: int, u: complex):
        alpha_j = alpha[i].reshape(-1, 1)
        alpha_k = alpha_j.T
        beta_j = beta[i].reshape(-1, 1)
        beta_k = beta_j.T
        envelope = jnp.exp(-0.5 * jnp.abs(u + beta_k - beta_j) ** 2)
        phase = jnp.exp(1j * (aOmegab(u, beta_k + beta_j) + aOmegab(beta_k, beta_j)))
        alpha_jk = alpha_j * jnp.conj(alpha_k)
        return (alpha_jk * envelope * phase).sum()

    @jax.jit
    def char_func_tot(u: complex):
        total = 0.0
        for i in range(alpha.shape[0]):
            total += char_func_i(i, u)
        return total

    return char_func_tot


@jax.jit
def analytic_fidelity_i(
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
    phase = jnp.exp(1j * aOmegab(da, betaj))
    return jnp.abs(jnp.sum(prefactor * envelope * phase)) ** 2


@jax.jit
def analytic_fidelity(
    all_coeffs_a: Array, all_coeffs_b: Array, all_peaks_a: Array, all_peaks_b: Array
):
    N = all_peaks_a.shape[0]
    M = all_peaks_b.shape[0]

    def body_i(i, acci):
        def body_j(j, accj):
            return accj + analytic_fidelity_i(
                coeffs_a=all_coeffs_a[i],
                coeffs_b=all_coeffs_b[j],
                peaks_a=all_peaks_a[i],
                peaks_b=all_peaks_b[j],
            )

        return jax.lax.fori_loop(0, M, body_j, acci)

    return jax.lax.fori_loop(0, N, body_i, 0.0)


@jax.jit
def analytic_fidelity_transfer_i(
    alpha_i: Array, beta_i: Array, c: Array, d: Array, cp: Array, dp: Array
):
    N = alpha_i.shape[0]
    A = c.shape[0]
    Ap = cp.shape[0]
    alpha_i = alpha_i.reshape((N, 1, 1))
    beta_i = beta_i.reshape((N, 1, 1))
    c = c.reshape((1, A, 1))
    d = d.reshape((1, A, 1))
    cp = cp.reshape((1, 1, Ap))
    dp = dp.reshape((1, 1, Ap))
    prefactor = alpha_i * c * jnp.conj(cp)
    exponential = jnp.exp(
        -0.5 * jnp.abs(beta_i - dp + d) ** 2
        + 1.0j * aOmegab(dp, beta_i)
        + 1.0j * aOmegab(d, beta_i - dp)
    )
    return jnp.abs(jnp.sum(prefactor * exponential)) ** 2


@jax.jit
def analytic_fidelity_transfer(
    alpha: Array, beta: Array, c: Array, d: Array, cp: Array, dp: Array
):
    def body_i(i, acc):
        return acc + analytic_fidelity_transfer_i(alpha[i], beta[i], c, d, cp, dp)

    return jax.lax.fori_loop(0, alpha.shape[0], body_i, 0.0)


@jax.jit
def analytic_pureloss_recovery_fidelity_thetaphi_iab(
    da: Array,
    db: Array,
    alpha_i: Array,
    beta_i: Array,
    cap: Array,
    dap: Array,
    gamma: float,
):
    A = cap.shape[0]
    N = alpha_i.shape[0]
    cap = cap.reshape(A, 1)
    dap = dap.reshape(A, 1)
    alpha_i = alpha_i.reshape(1, N)
    beta_i = beta_i.reshape(1, N)

    prefactor = jnp.conj(cap) * alpha_i
    env_term1 = (-1.0 + jnp.sqrt(1 - gamma)) / 2.0 * jnp.abs(beta_i - dap) ** 2
    env_term2 = (
        -jnp.sqrt(1 - gamma)
        / 2.0
        * (jnp.abs(beta_i - dap + db) ** 2 - jnp.abs(db) ** 2)
    )
    env_term3 = -0.25 * (
        gamma * jnp.abs(da - db) ** 2
        + (1.0 - gamma) * (jnp.abs(da) ** 2 + jnp.abs(db) ** 2)
    )
    envelope = env_term1 + env_term2 + env_term3
    phase = 1.0j * (
        aOmegab(dap, beta_i)
        + jnp.sqrt(1 - gamma) * aOmegab(db, beta_i - dap)
        + gamma / 2 * aOmegab(da, db)
    )

    return jnp.sum(prefactor * jnp.exp(envelope + phase))


@jax.jit
def analytic_pureloss_recovery_fidelity_thetaphi(
    alpha: Array, beta: Array, c: Array, d: Array, gamma: float
):
    # filthy, but lowers complexity from O(A^2) to O(A)
    partial_i_fidelity_caller = jax.jit(
        partial(
            jax.vmap(
                jax.vmap(
                    lambda da,
                    db,
                    alpha_i,
                    beta_i: analytic_pureloss_recovery_fidelity_thetaphi_iab(
                        da, db, alpha_i, beta_i, c, d, gamma
                    )
                    * jnp.conj(
                        analytic_pureloss_recovery_fidelity_thetaphi_iab(
                            db, da, alpha_i, beta_i, c, d, gamma
                        )
                    ),
                    in_axes=(None, 0, None, None),
                ),
                in_axes=(0, None, None, None),
            ),
            d,
            d,
        )
    )

    def body_i(i, acc):  # would have been vmapped but it's too big
        return acc + jnp.abs(
            jnp.einsum(
                "ij,i,j->", partial_i_fidelity_caller(alpha[i], beta[i]), jnp.conj(c), c
            )
        )

    return jax.lax.fori_loop(0, alpha.shape[0], body_i, 0.0)


@partial(jax.jit, static_argnums=2)
def analytic_fidelity_wrapper(
    coherent: CoherentKet, circuit_parameters: Array, N_l: int
):
    alpha_coherent = jnp.expand_dims(coherent.cs, 0)
    beta_coherent = jnp.expand_dims(coherent.ds, 0)

    alpha_circuit, beta_circuit = g(circuit_parameters, N_l)

    return analytic_fidelity(
        all_coeffs_a=alpha_coherent,
        all_coeffs_b=alpha_circuit,
        all_peaks_a=beta_coherent,
        all_peaks_b=beta_circuit,
    )


@partial(jax.jit, static_argnums=(2, 3))
def analytic_fidelity_multi_wrapper(
    coherent: CoherentKet, circuit_parameters: Array, N_l: int, T: int
):
    alpha_coherent = jnp.expand_dims(coherent.cs, 0)
    beta_coherent = jnp.expand_dims(coherent.ds, 0)

    alpha_circuit, beta_circuit = super_g(circuit_parameters, N_l=N_l, T=T)
    return analytic_fidelity(
        all_coeffs_a=alpha_coherent,
        all_coeffs_b=alpha_circuit,
        all_peaks_a=beta_coherent,
        all_peaks_b=beta_circuit,
    )


@partial(jax.jit, static_argnums=(3, 4))
def analytic_fidelity_transfer_wrapper(
    initial: CoherentKet, final: CoherentKet, circuit_params: Array, N_l: int, T: int
):
    alpha, beta = super_g(circuit_params, N_l=N_l, T=T)
    return analytic_fidelity_transfer(
        alpha=alpha, beta=beta, c=initial.cs, d=initial.ds, cp=final.cs, dp=final.ds
    )


@partial(jax.jit, static_argnums=(3, 4))
def analytic_pureloss_recovery_fidelity_random_wrapper(
    logical_0: CoherentKet,
    logical_1: CoherentKet,
    gamma: float,
    N_l: int,
    T: int,
    circuit_params: Array,
    key: jr.PRNGKey,
):
    alpha, beta = super_g(circuit_params, N_l=N_l, T=T)
    u = jr.uniform(key, (2,))
    theta = jnp.arccos(2 * u[0] - 1.0)
    phi = 2.0 * jnp.pi * u[1]
    c0 = jnp.cos(theta / 2)
    c1 = jnp.sin(theta / 2) * jnp.exp(1.0j * phi)
    cs = jnp.concatenate([c0 * logical_0.cs, c1 * logical_1.cs])
    ds = jnp.concat([logical_0.ds, logical_1.ds])
    return analytic_pureloss_recovery_fidelity_thetaphi(
        alpha=alpha, beta=beta, c=cs, d=ds, gamma=gamma
    )


def optimize_wrt_state(
    target_state: CoherentKet,
    N_depth=6,
    lr=0.005,
    steps=10000,
    restarts=5,
    random_dist=4.0,
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
            a = a.at[:, 3].set(
                jnp.zeros(
                    N_depth,
                )
            )  # kill the gammas
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


def optimize_wrt_superstate(
    target_state: CoherentKet,
    T_depth=2,
    N_depth=6,
    lr=0.005,
    steps=10000,
    restarts=5,
    random_dist=4.0,
):
    N_layer = 2 ** (N_depth)

    @partial(jax.jit, static_argnums=(0, 1))
    def analytic_loss_multi_fn(N_l: int, T: int, circuit_params: Array):
        return 1.0 - analytic_fidelity_multi_wrapper(
            target_state, circuit_params, N_l=N_l, T=T
        )

    @partial(jax.jit, static_argnums=(2, 3))
    def analytic_train_multi_step(a: Array, opt_state: Any, N_l: int, T: int):
        a = a.astype(jnp.complex64)
        partial_analytic_multi_loss_fn = partial(analytic_loss_multi_fn, N_l, T)
        grads = jax.grad(partial_analytic_multi_loss_fn)(a)
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
        a_init = a_init.at[:, :, 1:].set(
            2 * jnp.pi * jr.uniform(key=k2, shape=(T_depth, N_depth, 3))
        )
        a_init = a_init.at[:, :, 0].set(
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
            a, opt_state = analytic_train_multi_step(
                a=a, opt_state=opt_state, N_l=N_layer, T=T_depth
            )
            if step_i % 100 == 0:
                current_loss = analytic_loss_multi_fn(N_layer, T_depth, a)
                loss_vals = loss_vals.at[step_i].set(current_loss.item())
                print(
                    f"Restart {restart}, Step {step_i}, 1 - F = {current_loss.item():.6f}"
                )
                if last_loss == current_loss and current_loss > 0.05:
                    print(f"Ending restart {restart} early.")
                    # a = a.at[:, 1:].set(0.01 * jr.normal(key=k2, shape=(N_depth, 3)))
                    break
                last_loss = current_loss
        current_loss = analytic_loss_multi_fn(N_l=N_layer, T=T_depth, circuit_params=a)
        if current_loss < best_loss:
            best_loss = current_loss
            best_val = a
            print(restart, "new best", best_loss)
            print(best_val)

    return best_val, best_loss


# TODO: abstract all these optimizers up
def optimize_wrt_transfer(
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


def optimize_wrt_pureloss_recovery(
    logical_0: CoherentKet,
    logical_1: CoherentKet,
    gamma: float,
    T_depth=1,
    N_depth=6,
    lr=0.005,
    steps=10000,
    restarts=5,
    batch_size=350,
    random_dist=4.0,
    random_angle=1.0,
):
    N_layer = 2 ** (N_depth)
    t_key = jr.PRNGKey(np.random.randint(1000))

    @jax.jit
    def analytic_pureloss_recovery_fidelity_random_fn(
        circuit_params: Array,
        key: jr.PRNGKey,
    ):
        keys = jr.split(key, batch_size)

        fids = jax.vmap(
            partial(
                analytic_pureloss_recovery_fidelity_random_wrapper,
                logical_0,
                logical_1,
                gamma,
                N_layer,
                T_depth,
                circuit_params,
            )
        )(keys)
        return 1.0 - jnp.mean(fids)

    u = jr.uniform(t_key, (2, batch_size))
    thetas = jnp.arccos(2 * u[0] - 1.0)
    phis = 2.0 * jnp.pi * u[1]
    c0s = jnp.ones_like(thetas)  # jnp.cos(thetas / 2)
    c1s = jnp.zeros_like(thetas)  # jnp.sin(thetas / 2) * jnp.exp(1.0j * phis)

    @jax.jit
    def analytic_pureloss_recovery_fidelity_deterministic_fn(
        circuit_params: Array,
    ):
        alpha, beta = super_g(circuit_params, N_l=N_layer, T=T_depth)
        fid_fn = jax.jit(
            partial(analytic_pureloss_recovery_fidelity_thetaphi, alpha, beta)
        )

        @jax.jit
        def fid_cs_fn(c0, c1):
            return fid_fn(
                jnp.concatenate([c0 * logical_0.cs, c1 * logical_1.cs]),
                jnp.concat([logical_0.ds, logical_1.ds]),
                gamma,
            )

        return 1 - jnp.mean(jax.vmap(fid_cs_fn)(c0s, c1s))

    # @jax.jit
    def analytic_pureloss_recovery_random_train_step(
        a: Array, opt_state: Any, key: jr.PRNGKey
    ):
        a = a.astype(jnp.complex64)
        loss_fn = eqx.filter_value_and_grad(
            analytic_pureloss_recovery_fidelity_random_fn
        )
        loss, grads = loss_fn(a, key)
        key = jr.split(key, 1)[0]
        updates, new_opt_state = optimizer.update(jnp.conj(grads), opt_state)
        a_updated = optax.apply_updates(a, updates)
        return a_updated, new_opt_state, key, loss

    @jax.jit
    def analytic_pureloss_recovery_deterministic_train_step(a: Array, opt_state: Any):
        a = a.astype(jnp.complex64)
        loss_fn = eqx.filter_value_and_grad(
            analytic_pureloss_recovery_fidelity_deterministic_fn
        )
        loss, grads = loss_fn(a)
        updates, new_opt_state = optimizer.update(jnp.conj(grads), opt_state)
        a_updated = optax.apply_updates(a, updates)
        return a_updated, new_opt_state, loss

    best_loss = 1.0
    best_val = None
    loss_vals = jnp.zeros((steps,), dtype=jnp.float32)
    for restart in range(restarts):
        key = jr.PRNGKey(np.random.randint(1000))
        optimizer = optax.adam(lr)
        k1, k2, k3, key = jr.split(key, 4)
        a_init = jnp.zeros((T_depth, N_depth, 4), jnp.complex64)
        print(a_init.shape)
        a_init = a_init.at[:, :, 1:].set(
            2 * random_angle * jnp.pi * jr.uniform(key=k2, shape=(T_depth, N_depth, 3))
        )
        a_init = a_init.at[:, :, 0].set(
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
            # a, opt_state, key, current_loss = (
            #     analytic_pureloss_recovery_random_train_step(
            #         a=a, opt_state=opt_state, key=key
            #     )
            # )
            a, opt_state, current_loss = (
                analytic_pureloss_recovery_deterministic_train_step(
                    a=a, opt_state=opt_state
                )
            )
            if step_i % 10 == 0:
                loss_vals = loss_vals.at[step_i].set(current_loss.item())
                print(
                    f"Restart {restart}, Step {step_i}, 1 - F_avg = {current_loss.item():.6f}"
                )
                if last_loss == current_loss and current_loss > 0.05:
                    print(f"Ending restart {restart} early.")
                    # a = a.at[:, 1:].set(0.01 * jr.normal(key=k2, shape=(N_depth, 3)))
                    break
                last_loss = current_loss
        if current_loss < best_loss:
            best_loss = current_loss
            best_val = a
            print(restart, "new best", best_loss)
            print(best_val)

    return best_val, best_loss


def array_to_latex_table(a: Array) -> str:
    a = jnp.round(a, 3)
    d_vals = np.array(a[:, 0]).astype(complex)
    phi_vals = np.array(jnp.real(a[:, 1])).astype(float)
    theta_vals = np.array(jnp.real(a[:, 2])).astype(float)
    gamma_vals = np.array(jnp.real(a[:, 3])).astype(float)

    s = r"\begin{tabular}{|c|c|c|c|c|}" + "\n" + r"\hline" + "\n"
    s += "Layer & $d$ & $\\phi$ & $\\theta$ & $\\gamma$ \\\\" + "\n" + r"\hline" + "\n"
    for i in range(len(d_vals)):
        d = d_vals[i]
        if np.iscomplexobj(d):
            re = round(float(np.real(d)), 3)
            im = round(float(np.imag(d)), 3)
            d_str = (
                f"{re:.3f}"
                if im == 0.0
                else f"{re:.3f}{'+' if im >= 0 else '-'}{abs(im):.3f}i"
            )
        else:
            d_str = f"{float(d):.3f}"
        s += (
            f"{i + 1} & {d_str} & {phi_vals[i]:.3f} & {theta_vals[i]:.3f} & {gamma_vals[i]:.3f} \\\\"
            + "\n"
            + r"\hline"
            + "\n"
        )
    s += r"\end{tabular}"
    return s


def array_to_scaled_latex_table(a: Array) -> str:
    a = jnp.round(a, 3)
    d_vals = np.array(a[:, 0]).astype(complex)
    phi_vals = np.array(jnp.real(a[:, 1])).astype(float)
    theta_vals = np.array(jnp.real(a[:, 2])).astype(float)
    gamma_vals = np.array(jnp.real(a[:, 3])).astype(float)

    s = r"\begin{tabular}{|c|c|c|c|c|}" + "\n" + r"\hline" + "\n"
    s += (
        "Layer & $d$ & $\\phi/\\pi$ & $\\theta/\\pi$ & $\\gamma/\\pi$ \\\\"
        + "\n"
        + r"\hline"
        + "\n"
    )
    for i in range(len(d_vals)):
        d = d_vals[i]
        if np.iscomplexobj(d):
            re = round(float(np.real(d)), 3)
            im = round(float(np.imag(d)), 3)
            d_str = (
                f"{re:.3f}"
                if im == 0.0
                else f"{re:.3f}{'+' if im >= 0 else '-'}{abs(im):.3f}i"
            )
        else:
            d_str = f"{float(d):.3f}"
        s += (
            f"{i + 1} & {d_str} & {phi_vals[i] / jnp.pi:.3f} & {theta_vals[i] / jnp.pi:.3f} & {gamma_vals[i] / jnp.pi:.3f} \\\\"
            + "\n"
            + r"\hline"
            + "\n"
        )
    s += r"\end{tabular}"
    return s


def gkp_coherent_dm(
    mu: int,
    N_trunc: int,
    Delta: float,
    lattice: str = "rect",
    lam=jnp.sqrt(2.0),
    N_trunc_y: int = None,
):
    # Quantum Error Correction with the Gottesman-Kitaev-Preskill Code, Grimsmo & Puri
    if N_trunc_y is None:
        N_trunc_y = N_trunc
    if lattice == "square":
        GKP_alpha = jnp.sqrt(jnp.pi / 2)
        GKP_beta = 1.0j * jnp.sqrt(jnp.pi / 2)
    elif lattice == "rect":
        GKP_alpha = lam * jnp.sqrt(jnp.pi / 2)
        GKP_beta = 1.0j * jnp.sqrt(jnp.pi / 2) / lam
    else:
        raise NotImplementedError
    cs = []
    ds = []
    for k in range(-N_trunc, N_trunc + 1):
        for l in range(-N_trunc_y, N_trunc_y + 1):
            disp = (2 * k + mu) * GKP_alpha + l * GKP_beta
            cs.append(
                jnp.exp(
                    -1.0j * jnp.pi * (k * l + l * mu / 2.0)
                    - (Delta**2) * jnp.abs(disp) ** 2
                )
            )
            ds.append(disp)
    return CoherentKet(cs=jnp.array(cs), ds=jnp.array(ds))


def analytic_circuit_layer(beta, theta, phi):
    return (
        dqtensor(dqdisplace(GKP_N, -beta / 2), dqfock_dm(2, 0))
        * (
            2 * jnp.cos(phi / 2) * jnp.sin(phi / 2) * jnp.sin(theta / 2)
            - 1j * jnp.sin(theta / 2) * jnp.cos(phi)
        )
        + dqtensor(dqdisplace(GKP_N, beta / 2), dqfock_dm(2, 1))
        * (
            -2 * jnp.cos(phi / 2) * jnp.sin(phi / 2) * jnp.sin(theta / 2)
            - 1j * jnp.sin(theta / 2) * jnp.cos(phi)
        )
        + dqtensor(dqdisplace(GKP_N, beta / 2), ket1 @ dqdag(ket0)) * jnp.cos(theta / 2)
        + dqtensor(dqdisplace(GKP_N, -beta / 2), ket0 @ dqdag(ket1))
        * jnp.cos(theta / 2)
    )


def make_pureloss_fock(gamma: float, rank: int, N: int = GKP_N):
    n_hat = dqnumber(N)
    a_hat = dqdestroy(N)
    return jnp.array(
        [
            (gamma / (1 - gamma)) ** (l / 2)
            / jnp.sqrt(math.factorial(l))
            * jnp.linalg.matrix_power(a_hat, l)
            @ jla.expm(jnp.log(1 - gamma) * n_hat / 2)
            for l in range(rank)
        ]
    )


def make_transpose_for_pureloss(
    loss_ops_in, logical_0: CoherentKet, logical_1: CoherentKet, eps=1e-5
):
    P = logical_0.to_fock_basis() + logical_1.to_fock_basis()
    loss_P = apply_kraus_map_nonorm(loss_ops_in, P)
    loss_P_eigs, loss_P_vecs = jnp.linalg.eigh(loss_P)

    def supp_invsqrt(arr):
        return jnp.where(arr != 0, arr**-0.5, arr)

    loss_P_eigs2 = supp_invsqrt(jnp.round(loss_P_eigs, decimals=int(-jnp.log10(eps))))
    loss_P_invsqrt = loss_P_vecs @ jnp.diag(loss_P_eigs2) @ dqdag(loss_P_vecs)
    inv_loss_ops = jnp.array(
        [dqdag(loss_ops_in[i, :, :]) for i in range(loss_ops_in.shape[0])]
    )
    return jnp.array(
        [
            P @ inv_loss_ops[i, :, :] @ loss_P_invsqrt
            for i in range(loss_ops_in.shape[0])
        ]
    )


def gate_timer(beta):
    chi = 2 * jnp.pi * 5e4
    gamma_0 = 20
    return jnp.clip(jnp.abs(beta) / chi / gamma_0, min=48e-9)  # p37


ancilla_time = 24e-9


def circuit_params_to_time(circuit_params):
    T = 0
    if len(circuit_params.shape) == 2:
        circuit_params = jnp.expand_dims(circuit_params, 0)
    for i in range(circuit_params.shape[0]):
        for j in range(circuit_params.shape[1]):
            T += ancilla_time + gate_timer(jnp.array(circuit_params[i, j, 0]))
    return T
