import jax.numpy as jnp
import jax.random as jr
from typing import Sequence
from jaxtyping import Array
import jax
import dynamiqs as dq
import equinox as eqx


# TODO rejit everything


@jax.jit
def aOmegab(a: Array, b: Array):
    return jnp.real(a) * jnp.imag(b) - jnp.imag(a) * jnp.real(b)


@jax.jit
def dag(arr):
    return jnp.conj(arr).T


@jax.jit
def coherent_overlap(alpha: Array, beta: Array):
    # <alpha|beta>
    # alpha,beta should be broadcastable
    return jnp.exp(-0.5 * jnp.abs(alpha - beta) ** 2 + 1.0j * aOmegab(alpha, beta))


@jax.jit
def invsqrtm(A: Array):
    w, v = jnp.linalg.eigh(A)
    return (v / jnp.sqrt(w)) @ dag(v)


@jax.jit
def invsqrtm_supp(A: Array):
    w, V = jnp.linalg.eigh(A)
    s = jnp.where(w > 1e-6, 1.0 / jnp.sqrt(w), 0.0)
    return V @ (s[:, None] * dag(V))


@jax.jit
def unitary_encoding_map(X: Array, d: Array, psi_logical: Array):
    # C: D(... x H_D) -> D(... x H_B)
    # C(rho) = U_C rho U_C^{\dagger}
    # U_C = \sum_i |\psi_i^C x i |
    # |\psi_i^C> = \sum_a c_ai |d_ai>
    # |\Psi_L> = \sum_{\mu\nu} c_{\mu\nu} |\mu>_A |\nu>_B
    # X: (A,D)
    # d: (A,)
    # psi_logical: (...,D,)
    A = d.shape[0]
    G = coherent_overlap(d.reshape((A, 1)), d.reshape((1, A)))  # (A,A)
    Q = X @ invsqrtm(dag(X) @ X)  # (A,D)
    C = invsqrtm(G) @ Q  # (A,D)
    return jnp.einsum("...l,al->...a", psi_logical, C)


# @jax.jit
def cptp_decoding_map(Z_D: Array, d: Array, rho_logical: Array):
    # D: D(... x H_B) -> D(... x H_D)
    # D(rho) = \sum_i D_i rho D_i^{\dagger}
    # D_i = \sum_{ka}^{DA} A_{ika} |k x \phi^D_i|
    # rho_logical: (D,D,A,A)
    # Z_D: (N_D, D, A)
    # d: (A,)
    A = d.shape[0]
    S = jnp.einsum("ika,ikb->ab", jnp.conj(Z_D), Z_D)
    print("S", jnp.linalg.matrix_rank(S))
    Sn5 = invsqrtm_supp(S)
    B = jnp.einsum("ikb,ba->ika", Z_D, Sn5)

    G = coherent_overlap(d.reshape((A, 1)), d.reshape((1, A)))
    print("G rank", jnp.linalg.matrix_rank(G), G.shape)
    lambda_G, U_G = jnp.linalg.eigh(G)
    Tpinv = jnp.diag(lambda_G**0.5) @ dag(U_G)
    return jnp.einsum(
        "pki,ia,uvab,jb,plj->uvkl", B, Tpinv, rho_logical, jnp.conj(Tpinv), jnp.conj(B)
    )


# @jax.jit
# def beamsplit_pure(alpha: Array, beta: Array, gamma: Array, sigma: Array, eta: float):
#     A = alpha.shape[0]
#     N_E = gamma.shape[0]
#     beta = beta.reshape((A, 1))
#     sigma = sigma.reshape((1, N_E))
#     d_out = jnp.sqrt(eta) * beta + jnp.sqrt(1 - eta) * sigma  # (A, N_E)
#     d_mix = jnp.sqrt(eta) * sigma - jnp.sqrt(1 - eta) * beta  # (A, N_E)
#     d_ik = jnp.reshape(d_mix, (A, 1, N_E, 1))
#     d_jl = jnp.reshape(d_mix, (1, A, 1, N_E))
#     mixer = coherent_overlap(d_jl, d_ik)  # (A, A, N_E, N_E)
#     rho_out = (
#         mixer
#         * jnp.reshape(alpha, (A, 1, 1, 1))
#         * jnp.conj(jnp.reshape(alpha, (1, A, 1, 1)))
#         * jnp.reshape(gamma, (1, 1, N_E, 1))
#         * jnp.conj(jnp.reshape(gamma, (1, 1, 1, N_E)))
#     )
#     return rho_out.reshape((A * N_E, A * N_E)), d_out.reshape((A * N_E,))


@jax.jit
def beamsplit_full(alpha: Array, beta: Array, Y: Array, sigma: Array, eta: float):
    D, A = alpha.shape
    N_E = sigma.shape[0]
    beta = beta.reshape((A, 1))
    sigma = sigma.reshape((1, N_E))
    gamma = Y / jnp.sqrt(
        jnp.einsum(
            "i,ij,j->",
            jnp.conj(Y),
            coherent_overlap(sigma.reshape((N_E, 1)), sigma.reshape((1, N_E))),
            Y,
        )
    )
    d_out = jnp.sqrt(eta) * beta + jnp.sqrt(1 - eta) * sigma  # (A, N_E)
    d_mix = jnp.sqrt(eta) * sigma - jnp.sqrt(1 - eta) * beta  # (A, N_E)
    d_ik = jnp.reshape(d_mix, (A, 1, N_E, 1))
    d_jl = jnp.reshape(d_mix, (1, A, 1, N_E))
    mixer = coherent_overlap(d_jl, d_ik)  # (A, A, N_E, N_E)
    rho_out = (
        jnp.reshape(mixer, (1, 1, A, A, N_E, N_E))
        * jnp.reshape(alpha, (D, 1, A, 1, 1, 1))
        * jnp.conj(jnp.reshape(alpha, (1, D, 1, A, 1, 1)))
        * jnp.reshape(gamma, (1, 1, 1, 1, N_E, 1))
        * jnp.conj(jnp.reshape(gamma, (1, 1, 1, 1, 1, N_E)))
    )
    rho_out = jnp.einsum("ijabmn->ijam bn", rho_out, optimize=True).reshape(
        D, D, A * N_E, A * N_E
    )
    return rho_out, d_out.reshape((A * N_E,))  # (D,D,A N_E,A N_E), (A N_E,)


def ket(D, i):
    return dq.fock(D, i).to_jax().flatten()


def complex_normal(key: jr.PRNGKey, shape: Sequence[int]):
    kr, ki = jr.split(key, 2)
    return jr.normal(kr, shape) + 1.0j * jr.normal(ki, shape)


class Channel(eqx.Module):
    X: Array
    d: Array
    Y: Array
    sigma: Array
    Z: Array

    def __init__(self, X, d, Y, sigma, Z):
        self.X = X
        self.d = d
        self.Y = Y
        self.sigma = sigma
        self.Z = Z

    def conj(self):
        return Channel(
            jnp.conj(self.X),
            jnp.conj(self.d),
            jnp.conj(self.Y),
            jnp.conj(self.sigma),
            jnp.conj(self.Z),
        )

    def __str__(self):
        return (
            str(f"{self.X=}")
            + str(f"{self.d=}")
            + str(f"{self.Y=}")
            + str(f"{self.sigma=}")
            + str(f"{self.Z=}")
        )

    def valid(self):
        return [
            bool(jnp.all(~jnp.isnan(self.X))),
            bool(jnp.all(~jnp.isnan(self.d))),
            bool(jnp.all(~jnp.isnan(self.Y))),
            bool(jnp.all(~jnp.isnan(self.sigma))),
            bool(jnp.all(~jnp.isnan(self.Z))),
        ]

    def jitter(self, key, scale: float):
        k1, k2, k3, k4, k5 = jr.split(key, 5)
        return Channel(
            self.X * (1 + scale * complex_normal(k1, self.X.shape)),
            self.d * (1 + scale * complex_normal(k2, self.d.shape)),
            self.Y * (1 + scale * complex_normal(k3, self.Y.shape)),
            self.sigma * (1 + scale * complex_normal(k4, self.sigma.shape)),
            self.Z * (1 + scale * complex_normal(k5, self.Z.shape)),
        )
