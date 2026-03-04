import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from functools import partial
from gkp_utils.utils import GKP_N, dqdisplace, dqdag, dqeye
from jaxtyping import Array
import equinox as eqx
import numpy as np
import optax

"""
n = circuit depth
m = # of traceouts

    A^m_n = (C^n, R^n, R^n) (circuit params)
        A_n.shape = (2^m,n,3)

    B^l_p = (C^p, C^p) aka the points in char func space
        B_p.shape = (l,p,2)

    g: A_n -> B_{2^n}
        the analytic compose function from circuit parameters to the space of points in charfunc space

    F: B_p1 x B_p2 -> R
        fidelity function between two char funcs
"""
N_depth = 10  # TODO rewrite more generally
N_layer = 2 ** (N_depth - 1)


# <Array Manipulation Utils>
@partial(jax.jit, static_argnums=1)
def mask(n: int, N_d):
    # generates a sequence with 2^n of each {1}^n etc
    return 1.0 + 2.0 * jnp.floor(0.999 * jnp.sin(jnp.pi * jnp.arange(0, N_d) / (2**n)))


@partial(jax.jit, static_argnums=1)
def addmask(n: int, N_l: int):
    index = 2**n
    mask = jnp.arange(N_l) < index
    return mask.astype(jnp.complex64)


@partial(jax.jit, static_argnums=1)
def caddmask(n: int, N_l: int):
    return jnp.roll(addmask(n, N_l), 2**n, axis=0)


@jax.jit
def dphase(alpha: Array, beta: Array):
    return jnp.exp((alpha * jnp.conj(beta) - jnp.conj(alpha) * beta) / 2.0)


# </Array Manipulation Utils>


# JAX utilities for manipulating characteristic function in an autodiff compatible way


@jax.jit
def N_l_from_N_d(N_d: int) -> int:
    return 2 ** (N_d - 1)


class JLayer(eqx.Module):
    pass


class JLayer(eqx.Module):
    n: int
    N_l: int
    coeffs: Array
    disps: Array

    def __init__(
        self,
        nin: int,
        N_l: int,
        coeffs: Array | None = None,
        disps: Array | None = None,
    ):
        self.n = nin
        self.N_l = N_l
        if coeffs is None:
            self.coeffs = jnp.zeros((2, 2, N_l), dtype=jnp.complex64)
        else:
            self.coeffs = coeffs
        if disps is None:
            self.disps = jnp.zeros((2, 2, N_l), dtype=jnp.complex64)
        else:
            self.disps = disps

    @staticmethod
    def from_single_param(param: Array, N_l) -> JLayer:
        beta, phi, theta = param[0], jnp.real(param[1]), jnp.real(param[2])
        ncoeffs = jnp.zeros((2, 2, N_l), dtype=jnp.complex64)
        ndisps = jnp.zeros((2, 2, N_l), dtype=jnp.complex64)
        ncoeffs = ncoeffs.at[:, :, 0].set(
            jnp.array(
                [
                    [
                        2 * jnp.cos(phi / 2) * jnp.sin(phi / 2) * jnp.sin(theta / 2)
                        - 1j * jnp.sin(theta / 2) * jnp.cos(phi),
                        jnp.cos(theta / 2),
                    ],
                    [
                        jnp.cos(theta / 2),
                        -2 * jnp.cos(phi / 2) * jnp.sin(phi / 2) * jnp.sin(theta / 2)
                        - 1j * jnp.sin(theta / 2) * jnp.cos(phi),
                    ],
                ]
            )
        )
        ndisps = ndisps.at[:, :, 0].set(
            jnp.array([[-beta / 2, -beta / 2], [beta / 2, beta / 2]])
        )
        return JLayer(nin=1, N_l=N_l, coeffs=ncoeffs, disps=ndisps)

    @staticmethod
    @partial(jax.jit, static_argnums=2)
    def compose(jl_a: JLayer, jl_b: JLayer, N_l: int):
        # assumes jl_a pure Layer
        n_b: int = jl_b.n - 1
        n_add = addmask(n_b, N_l)
        n_cadd = caddmask(n_b, N_l)
        full_add = addmask(n_b + 1, N_l)
        cshift = lambda x: jnp.roll(x, 2**n_b, axis=0)

        drow0 = jl_a.disps[0, 0, 0] * full_add
        # TODO modify if supercomposing bc no longer same across row, and all further references where jl_a.x[_,_,y] has a concrete y
        drow1 = jl_a.disps[1, 1, 0] * full_add

        dcol0 = n_add * jl_b.disps[0, 0, :] + n_cadd * cshift(jl_b.disps[1, 0, :])
        dcol1 = n_add * jl_b.disps[0, 1, :] + n_cadd * cshift(jl_b.disps[1, 1, :])

        # print("nadd")
        # print(n_add)
        # print("cadd")
        # print(n_cadd)
        # print("fadd")
        # print(full_add)
        # print("rows")
        # print(drow0)
        # print(drow1)
        # print("cols")
        # print(dcol0)
        # print(dcol1)

        # TODO: rewrite this using the formula on p52

        out_disps = jnp.array(
            [[drow0 + dcol0, drow0 + dcol1], [drow1 + dcol0, drow1 + dcol1]]
        )

        c00 = (
            jl_a.coeffs[0, 0, 0]
            * n_add
            * jl_b.coeffs[0, 0, :]
            * dphase(jl_a.disps[0, 0, 0] * full_add, n_add * jl_b.disps[0, 0, :])
        ) + cshift(
            jl_a.coeffs[0, 1, 0]
            * n_add
            * jl_b.coeffs[1, 0, :]
            * dphase(jl_a.disps[0, 1, 0] * n_add, jl_b.disps[1, 0, :])
        )

        c01 = (
            jl_a.coeffs[0, 0, 0]
            * n_add
            * jl_b.coeffs[0, 1, :]
            * dphase(jl_a.disps[0, 0, 0] * full_add, n_add * jl_b.disps[0, 1, :])
        ) + cshift(
            jl_a.coeffs[0, 1, 0]
            * n_add
            * jl_b.coeffs[1, 1, :]
            * dphase(jl_a.disps[0, 1, 0] * n_add, jl_b.disps[1, 1, :])
        )

        c10 = (
            jl_a.coeffs[1, 0, 0]
            * n_add
            * jl_b.coeffs[0, 0, :]
            * dphase(jl_a.disps[1, 0, 0] * full_add, n_add * jl_b.disps[0, 0, :])
        ) + cshift(
            jl_a.coeffs[1, 1, 0]
            * n_add
            * jl_b.coeffs[1, 0, :]
            * dphase(jl_a.disps[1, 1, 0] * n_add, jl_b.disps[1, 0, :])
        )

        c11 = (
            jl_a.coeffs[1, 0, 0]
            * n_add
            * jl_b.coeffs[0, 1, :]
            * dphase(jl_a.disps[1, 0, 0] * full_add, n_add * jl_b.disps[0, 1, :])
        ) + cshift(
            jl_a.coeffs[1, 1, 0]
            * n_add
            * jl_b.coeffs[1, 1, :]
            * dphase(jl_a.disps[1, 1, 0] * n_add, jl_b.disps[1, 1, :])
        )

        out_coeffs = jnp.array([[c00, c01], [c10, c11]], dtype=jnp.complex64)

        return JLayer(
            nin=jl_b.n + 1, N_l=jl_a.coeffs.shape[2], coeffs=out_coeffs, disps=out_disps
        )

    @staticmethod
    @partial(jax.jit, static_argnums=1)
    def from_params(params: Array, N_l: int) -> JLayer:
        # this is the g function from the formalization
        circuit = JLayer.from_single_param(params[0, :], N_l)

        @jax.jit
        def body_compose(i: int, circ):
            return JLayer.compose(
                JLayer.from_single_param(params[i, :], N_l), circ, N_l
            )

        return jax.lax.fori_loop(
            1, params.shape[0], body_fun=body_compose, init_val=circuit
        )

    @staticmethod
    def from_symbolic(symbolic: np.ndarray, N_d: int) -> JLayer:
        N_l = N_l_from_N_d(N_d)
        coeffs = jnp.zeros((2, 2, N_l), dtype=jnp.complex64)
        disps = jnp.zeros((2, 2, N_l), dtype=jnp.complex64)
        for i in [0, 1]:
            for j in [0, 1]:
                for k, op in enumerate(symbolic[i, j].ops):
                    coeffs = coeffs.at[i, j, k].set(op.coeff)
                    disps = disps.at[i, j, k].set(op.alpha)
        return JLayer(nin=N_d, N_l=coeffs.shape[2], coeffs=coeffs, disps=disps)

    @staticmethod
    @jax.jit
    def to_traceout(jl: JLayer) -> Array:
        # (2,2,p) complex array of operators 0,1, alpha, beta (coeffs, displacements)
        # E_{\mu} = <\mu|\Pi U_i |0>
        return jnp.stack([jl.coeffs[:, 0, :], jl.disps[:, 0, :]], dtype=jnp.complex64)


@partial(jax.jit, static_argnums=1)
def g(params: Array, N_l: int) -> Array:
    # KEY: N_l should be a power of 2. If you have n displacements and a parameter shape of (n+1,3), you should input 2^{n} and it will update the alphas but not introduce redundant betas (ie single cycle sbs should be 8)
    return JLayer.to_traceout(JLayer.from_params(params, N_l))


@jax.jit
def choi_fidelity(b0, b1, sigma0=0.25, sigma1=0.25):
    alpha0 = b0[0, :, :]
    beta0 = b0[1, :, :]
    alpha1 = b1[0, :, :]
    beta1 = b1[1, :, :]

    alpha_outer = alpha0[:, :, None, None] * jnp.conj(alpha1[None, None, :, :])

    beta_diff = beta0[:, :, None, None] - beta1[None, None, :, :]

    dist_sq = jnp.abs(beta_diff) ** 2
    denom = sigma0**2 + sigma1**2
    factor = 2.0 * sigma0 * sigma1 / denom
    exponent = jnp.exp(-dist_sq / (2.0 * denom))
    # print(exponent)

    overlap_matrix = alpha_outer * factor * exponent

    return jnp.sum(jnp.abs(jnp.sum(overlap_matrix, axis=(1, 3))) ** 2, axis=(0, 1))


# @partial(jax.jit, static_argnums=[2,4])
def optimizer(b_star, a_init, steps=1000, lr=1e-3, plot_loss=False):
    # a* = argmin_a -F(g(a), b_star)

    @jax.jit
    def loss_fn(a):
        return 1000 * (1 - choi_fidelity(g(a), b_star))

    # - .1*jnp.abs(jnp.trace(net_qubit_action(a)))**2

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(a_init)

    @jax.jit
    def train_step(a, opt_state):
        grads = jax.grad(loss_fn)(a)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        a_updated = optax.apply_updates(a, updates)
        return a_updated, new_opt_state

    a = a_init
    loss_vals = jnp.zeros((steps,), dtype="float32")

    if not plot_loss:
        a, opt_state = jax.lax.fori_loop(
            0,  # start
            steps,  # end
            lambda i, carry: train_step(*carry),  # body function
            (a, opt_state),  # initial carry
        )
    else:
        for step_i in range(steps):
            a, opt_state = train_step(a, opt_state)
            if step_i % 1000 == 0:
                current_loss = loss_fn(a)
                loss_vals = loss_vals.at[step_i].set(current_loss.item())
                print(f"Step {step_i}, Loss = {current_loss.item():.6f}")

    return a, loss_vals


def display_alpha_beta(b, intermediates=False):
    # b (2,m,n)
    x_all = np.array([])
    y_all = np.array([])
    cmag_all = np.array([])
    cphase_all = np.array([])
    for i in range(b.shape[1]):
        xs = jnp.real(b[1, i, :])
        ys = jnp.imag(b[1, i, :])
        cms = jnp.abs(b[0, i, :])
        cps = jnp.angle(b[0, i, :])
        x_all = np.concatenate((x_all, xs))
        y_all = np.concatenate((y_all, ys))
        cmag_all = np.concatenate((cmag_all, cms))
        cphase_all = np.concatenate((cphase_all, cps))
        max_mag = np.max(cms)
        if intermediates:
            fig, ax = plt.subplots()
            cmap = plt.cm.hsv
            norm = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)
            scatter = ax.scatter(
                xs, ys, c=cps, s=cms / max_mag * 40, cmap=cmap, norm=norm
            )
            cbar = plt.colorbar(scatter, ax=ax, label="Phase (radians)")
            cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            cbar.set_ticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
            ax.set_xlabel("Real")
            ax.set_ylabel("Imag")
            ax.set_title(f"Characteristic Function Dirac Weights ({i})")
            plt.show()
    plt.clf()
    max_mag = np.max(cmag_all)
    fig, ax = plt.subplots()
    cmap = plt.cm.hsv
    norm = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)
    scatter = ax.scatter(
        x_all, y_all, c=cphase_all, s=40 * cmag_all / max_mag, cmap=cmap, norm=norm
    )
    cbar = plt.colorbar(scatter, ax=ax, label="Phase (radians)")
    cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar.set_ticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.set_title("Characteristic Function Dirac Weights All")
    plt.grid(True)
    plt.show()


# def display_alpha_beta_smoothed(b, intermediates=False, sigma=.2, grid_size= 100):
# b (2,m,n)
# x_all = np.array([])
# y_all = np.array([])
# cmag_all = np.array([])
# cphase_all = np.array([])
# for i in range(b.shape[1]):
#     xs = jnp.real(b[1,i,:])
#     ys = jnp.imag(b[1,i,:])
#     cms = jnp.abs(b[0,i,:])
#     cps = jnp.angle(b[0,i,:])
#     x_all = np.concatenate((x_all,xs))
#     y_all = np.concatenate((y_all,ys))
#     cmag_all = np.concatenate((cmag_all,cms))
#     cphase_all = np.concatenate((cphase_all,cps))
#     if intermediates:
#         fig, ax = plt.subplots()
#         x_min, x_max = float(jnp.min(xs)), float(jnp.max(xs))
#         y_min, y_max = float(jnp.min(ys)), float(jnp.max(ys))
#         m = 3*sigma
#         x_min -= m
#         x_max += m
#         y_min -= m
#         y_max += m
#         xg = jnp.linspace(x_min, x_max, grid_size)
#         yg = jnp.linspace(y_min, y_max, grid_size)
#         X, Y = jnp.meshgrid(xg, yg)
#         w = jnp.zeros((grid_size, grid_size), dtype=jnp.complex64)
#         for j in range(len(cms)):
#             cx, cy, amp = xs[j], ys[j], cms[j]
#             w = w + amp*jnp.exp(-((X-cx)**2 + (Y-cy)**2)/(2*sigma**2))
#         plt.clf()
#         ax.set_xlabel("Real")
#         ax.set_ylabel("Imag")
#         ax.set_title(f"Characteristic Function Dirac Weights ({i})")
#         ax.imshow(w, aspect='auto',extent=(x_min,x_max,y_min,y_max))
#         nm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)
#         sm = plt.cm.ScalarMappable(cmap=plt.cm.twilight, norm=nm)
#         sm.set_array([])
#         cb = plt.colorbar(sm, ax=ax)
#         cb.set_ticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
#         cb.set_ticklabels([r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$'])
#         plt.show()
# plt.clf()
# max_mag = np.max(cmag_all)
# fig, ax = plt.subplots()
# cmap = plt.cm.viridis
# norm = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)
# scatter = ax.scatter(x_all, y_all, c=cphase_all, s=40*cmag_all/max_mag, cmap=cmap, norm=norm)
# cbar = plt.colorbar(scatter, ax=ax, label='Phase (radians)')
# cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
# cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
# ax.set_xlabel("Real")
# ax.set_ylabel("Imag")
# ax.set_title("Characteristic Function Dirac Weights All")
# plt.grid(True)
# plt.show()


# def channel_from_b(b, generate_complement=False):
#     ops = jnp.zeros((b.shape[1],GKP_N,GKP_N))
#     for j in range(b.shape[1]):
#         ops = ops.at[j,:,:].set(jnp.sum(jnp.array([b[0,j,i]*dq.displace(GKP_N,b[1,j,i]) for i in range(b.shape[2])]),axis=0))
#     if generate_complement:
#         ops = ops.at[1,:,:].set(dq.eye(GKP_N)-dq.dag(ops[0,:,:])@ops[0,:,:])
#     return ops


def channel_from_b(b, generate_complement=False):
    ops = jnp.zeros((b.shape[1], GKP_N, GKP_N), dtype=jnp.complex64)

    @jax.jit
    def sum_displacements_over_i(j):
        @jax.jit
        def body_fun(i, partial_sum):
            return partial_sum + b[0, j, i] * dqdisplace(GKP_N, b[1, j, i]).astype(
                jnp.complex64
            )

        return jax.lax.fori_loop(
            0, b.shape[2], body_fun, jnp.zeros((GKP_N, GKP_N), dtype=jnp.complex64)
        )

    @jax.jit
    def outer_body_fun(j, ops_accum):
        return ops_accum.at[j, :, :].set(sum_displacements_over_i(j))

    ops = jax.lax.fori_loop(0, b.shape[1], outer_body_fun, ops)

    if generate_complement:
        ops = ops.at[1, :, :].set(dqeye(GKP_N) - dqdag(ops[0, :, :]) @ ops[0, :, :])

    return ops


@jax.jit
def e_ipiphi_vec(a, b):
    a_r = jnp.real(a)
    a_i = jnp.imag(a)
    b_r = jnp.real(b)
    b_i = jnp.imag(b)
    # Symplectic form: a \Omega b = (a_r * b_i) - (a_i * b_r)
    symplectic_val = a_r * b_i - a_i * b_r
    return jnp.exp(1j * jnp.pi * symplectic_val)


@jax.jit
def traceout_compose(b1: Array, b2: Array):
    alpha_mu = b1[0]  # shape (N_mu, K_mu)
    beta_mu = b1[1]  # shape (N_mu, K_mu)

    alpha_nu = b2[0]  # shape (N_nu, K_nu)
    beta_nu = b2[1]  # shape (N_nu, K_nu)

    N_mu, K_mu = alpha_mu.shape
    N_nu, K_nu = alpha_nu.shape

    alpha_mu_expanded = alpha_mu[:, None, :, None]
    beta_mu_expanded = beta_mu[:, None, :, None]
    alpha_nu_expanded = alpha_nu[None, :, None, :]
    beta_nu_expanded = beta_nu[None, :, None, :]

    new_alpha = (
        alpha_mu_expanded
        * alpha_nu_expanded
        * e_ipiphi_vec(beta_nu_expanded, beta_mu_expanded)
    )
    new_beta = beta_mu_expanded + beta_nu_expanded

    new_beta = jnp.reshape(new_beta, (N_mu * N_nu, K_mu * K_nu))

    new_alpha = jnp.reshape(new_alpha, (N_mu * N_nu, K_mu * K_nu))
    b12 = jnp.stack([new_alpha, new_beta], axis=0)
    return b12
