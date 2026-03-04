import os
import pickle
import scipy.io as sio
import jax
import sympy as sp
from sympy import sqrt
import numpy as np
import jax.numpy as jnp

pi = jnp.pi


def load_or_compute(expr, remap, fn):
    if os.path.exists(fn):
        with open(fn, "rb") as f:
            return pickle.load(f)

    val = sp.simplify(expr.subs(remap))
    with open(fn, "wb") as f:
        pickle.dump(val, f)
    return val


@jax.jit
def coth(x):
    return jnp.cosh(x) / jnp.sinh(x)


@jax.jit
def k_Delta(Deltav):
    return np.pi / 2 * coth(Deltav**2 / 2)


@jax.jit
def kp_Delta(k_Deltav):
    return k_Deltav / 2 + pi**2 / (8 * k_Deltav)


@jax.jit
def kpp_Delta(k_Deltav):
    return k_Deltav / 2 - pi**2 / (8 * k_Deltav)


@jax.jit
def k_gamma(gammav):
    return ((1 + jnp.sqrt(1 - gammav)) ** 2) / gammav


@jax.jit
def kp_gamma(k_gammav):
    return 2 * pi * k_gammav / ((1 + k_gammav) ** 2)


@jax.jit
def a_gamma(k_gammav):
    return (1 - k_gammav) / (1 + k_gammav)


@jax.jit
def B_gamma(k_gammav):
    return jnp.sqrt(1.0 + a_gamma(k_gammav) ** 2)


@jax.jit
def K(k_Deltav, k_gammav):
    return kp_Delta(k_Deltav) * (1 + a_gamma(k_gammav) ** 2) + kp_gamma(k_gammav)


@jax.jit
def n_bar(Deltav):
    return 1 / Deltav**2 - 0.5


@jax.jit
def Delta_from_nbar(n_barv):
    return 1.0 / jnp.sqrt(2 * n_barv + 1)


def sym_matrix_from_string_array(cell):
    arr = np.array(cell)
    rows, cols = arr.shape if arr.ndim == 2 else (len(arr), 1)
    if arr.ndim == 1:
        arr = arr.reshape(rows, 1)
    return sp.Matrix(
        [[sp.sympify(arr[i, j]) for j in range(cols)] for i in range(rows)]
    )


file_name = "../../eig_data_sympy.mat"
verbosity = 0

d = sio.loadmat(file_name, squeeze_me=True)

if verbosity > 0:
    print(d.keys())

E = sym_matrix_from_string_array(d["Ee_str"])
V = sym_matrix_from_string_array(d["Ve_str"])
b = sym_matrix_from_string_array(d["be_str"])
c = sp.sympify(d["ce_str"])
p = sp.Symbol("p")
pi_sym = sp.pi
I_sym = sp.I

all_exprs = [E, V, b, c]
all_symbols = set().union(*(expr.free_symbols for expr in all_exprs))

kappa_D_r = sp.Symbol("kappa_D_r", real=True, positive=True)
kappa_gamma_r = sp.Symbol("kappa_gamma_r", real=True, positive=True)
# mu_r = sp.Symbol('mu_r', real=True, positive=True, integer=True)
# nu_r = sp.Symbol('nu_r', real=True, positive=True, integer=True)
# mu_p_r = sp.Symbol('mu_p_r', real=True, positive=True, integer=True)
# nu_p_r = sp.Symbol('nu_p_r', real=True, positive=True, integer=True)
p_01 = sp.Symbol("p_01", real=True, positive=True, integer=True)
q_01 = sp.Symbol("q_01", real=True, positive=True, integer=True)
b_gamma_r = sp.Symbol("b_gamma_r", real=True, positive=True)
beta_jr, beta_ji, beta_kr, beta_ki = sp.symbols(
    "beta_jr, beta_ji, beta_kr, beta_ki", real=True
)
S_jkr, S_jki = sp.symbols("S_jkr, S_jki", real=True)
Delta_jkr, Delta_jki = sp.symbols("Delta_jkr, Delta_jki", real=True)

symbol_remap = {
    "p": pi_sym,
    "kappa_D": kappa_D_r,
    "kappa_gamma": kappa_gamma_r,
    # "mu": mu_r,
    # "nu": nu_r,
    # "mu_p": mu_p_r,
    # "nu_p": nu_p_r,
    "p_log": p_01,
    "q_log": q_01,
    # "beta_j1": beta_jr,
    # "beta_j2": beta_ji,
    # "beta_k1": beta_kr,
    # "beta_k2": beta_ki,
    "S_jk1": S_jkr,
    "S_jk2": S_jki,
    "Delta_jk1": Delta_jkr,
    "Delta_jk2": Delta_jki,
}

E_a = E.subs(symbol_remap)
V_a = V.subs(symbol_remap)

# b_a = sp.simplify(b.subs(symbol_remap))
# c_a = sp.simplify(c.subs(symbol_remap))
b_a = load_or_compute(b, symbol_remap, "b_a.pkl")
c_a = load_or_compute(c, symbol_remap, "c_a.pkl")

# Ev = E_a.subs({kappa_D_r: k_Deltav, kappa_gamma_r: k_gammav}, simultaneous=True).evalf() # TODO let verify these

alpha_subspace = [2, 3, 6, 7]  # small
beta_subspace = [0, 1, 4, 5]  # big

sigma_alpha = V_a[:, alpha_subspace]
s_eigs_alpha = sp.Matrix([E_a[alpha_i] for alpha_i in alpha_subspace])
# pprint("Alpha Subspace")
# for alpha_i in alpha_subspace:
#     pprint(E_a[alpha_i])
#     print(Ev[alpha_i])

rho_beta = V_a[:, beta_subspace]
l_eigs_beta = sp.Matrix([E_a[beta_i] for beta_i in beta_subspace])
# pprint("Beta Subspace")
# for beta_i in beta_subspace:
#     pprint(E_a[beta_i])
#     print(Ev[beta_i])

sqrt2 = sqrt(2)
c_alpha = sp.Matrix([sqrt2, sqrt2, b_gamma_r, b_gamma_r])
c_alpha2 = c_alpha.multiply_elementwise(c_alpha)
d_beta = sp.Matrix([sqrt2, sqrt2, b_gamma_r, b_gamma_r])
d_beta2 = d_beta.multiply_elementwise(d_beta)

phi_alpha = sp.simplify(sigma_alpha.T @ b_a)
psi_beta = sp.simplify(rho_beta.T @ b_a)

# Lambdification
condensed_swap_set = [kappa_D_r, kappa_gamma_r, b_gamma_r]
full_swap_set = [
    kappa_D_r,
    kappa_gamma_r,
    b_gamma_r,
    # beta_jr,
    # beta_ji,
    # beta_kr,
    # beta_ki,
    Delta_jkr,
    Delta_jki,
    S_jkr,
    S_jki,
    p_01,
    q_01,
    # mu_r,
    # mu_p_r,
    # nu_r,
    # nu_p_r,
]

# eval_s_eigs_alpha = jax.jit(sp.lambdify(condensed_swap_set, s_eigs_alpha, modules="jax"))
eval_jacobi_lambda_alpha = jax.jit(
    sp.lambdify(
        condensed_swap_set, s_eigs_alpha.multiply_elementwise(c_alpha2), modules="jax"
    )
)
# eval_jacobi_lambda_beta = jax.jit(sp.lambdify(condensed_swap_set, l_eigs_beta.multiply_elementwise(d_beta2), modules="jax"))

# jacobi_b_alpha = phi_alpha.multiply_elementwise(c_alpha)/(2*I_sym)
# jacobi_b_alpha_r, jacobi_b_alpha_i = jacobi_b_alpha.as_real_imag()
# eval_jacobi_b_alpha_r = jax.jit(sp.lambdify(full_swap_set, jacobi_b_alpha_r, modules="jax"))
# eval_jacobi_b_alpha_i = jax.jit(sp.lambdify(full_swap_set, jacobi_b_alpha_i, modules="jax"))

# @jax.jit
# def eval_jacobi_b_alpha(**kwargs):
#     return eval_jacobi_b_alpha_r(**kwargs) + 1.0j*eval_jacobi_b_alpha_i(**kwargs)

# jacobi_b_beta = psi_beta.multiply_elementwise(d_beta)/(2*I_sym)
# jacobi_b_beta_r, jacobi_b_beta_i = jacobi_b_beta.as_real_imag()
# eval_jacobi_b_beta_r = jax.jit(sp.lambdify(full_swap_set, jacobi_b_beta_r, modules="jax"))
# eval_jacobi_b_beta_i = jax.jit(sp.lambdify(full_swap_set, jacobi_b_beta_i, modules="jax"))

# @jax.jit
# def eval_jacobi_b_beta(**kwargs):
#     return eval_jacobi_b_beta_r(**kwargs) + 1.0j*eval_jacobi_b_beta_i(**kwargs)

eval_exp_c = jax.jit(sp.lambdify(full_swap_set, sp.exp(c_a), modules="jax"))

prod_term = sp.exp(
    sum((phi_alpha[i] ** 2) / (-4 * (s_eigs_alpha[i])) for i in range(4))
)
eval_prod_term = jax.jit(sp.lambdify(full_swap_set, prod_term, modules="jax"))


def verify_constants(Deltav, gammav):
    k_Deltav = k_Delta(Deltav)
    kp_Deltav = kp_Delta(k_Deltav)
    kpp_Deltav = kpp_Delta(k_Deltav)
    k_gammav = k_gamma(gammav)
    kp_gammav = kp_gamma(gammav)
    a_gammav = a_gamma(k_gammav)
    B_gammav = B_gamma(k_gammav)
    kpp_gammav = kp_gammav / a_gammav
    k_Delta_gammav = kp_Deltav * a_gammav + kpp_gammav
    Kv = K(k_Deltav, k_gammav)
    lambda_1v = -(kpp_Deltav**2 * B_gammav**2 - Kv * kp_Deltav) / (2 * Kv)

    print(rf"$k_{{\Delta}} = {k_Deltav:.3f}$")
    print(rf"a_{{\gamma}} = {a_gammav:.3f}")
    print(rf"b_{{\gamma}} = {B_gammav:.3f}")
    print(rf"k_{{\gamma}} = {k_gammav:.3f}")
    print(rf"k'_{{\gamma}} = {kp_gammav:.3f}")
    print(rf"k''_{{\gamma}} = {kpp_gammav:.3f}")
    print(rf"k'_{{\Delta \gamma}} = {k_Delta_gammav:.3f}")
    print(rf"K = {Kv:.3f}")
    print(rf"\lambda_1 = {lambda_1v:.3f}")


@jax.jit
def compute_prefactor(**kwargs):
    return (
        (jnp.pi**4)
        / (
            8
            * kp_Delta(k_Deltav=kwargs["kappa_D_r"])
            * K(k_Deltav=kwargs["kappa_D_r"], k_gammav=kwargs["kappa_gamma_r"])
        )
        * jnp.prod(jnp.sqrt(1 / jnp.abs(eval_jacobi_lambda_alpha(**kwargs))))
    )


# TODO jax.jit
def execute_numerics(**global_kwargs):
    global_kwargs_small = {  # TODO come up with better way to do this
        "kappa_D_r": global_kwargs["kappa_D_r"],
        "kappa_gamma_r": global_kwargs["kappa_gamma_r"],
        "b_gamma_r": global_kwargs["b_gamma_r"],
    }

    logicals = [0, 1]
    total = 0.0
    for mu in logicals:
        for nu in logicals:
            for mu_p in logicals:
                for nu_p in logicals:
                    global_kwargs["p_01"] = jnp.abs(nu - mu)
                    global_kwargs["q_01"] = jnp.abs(nu_p - mu_p)  # TODO fix this
                    exp_c = eval_exp_c(**global_kwargs)
                    total += exp_c * eval_prod_term(**global_kwargs)
    return total * compute_prefactor(**global_kwargs_small)
