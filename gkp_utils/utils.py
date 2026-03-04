import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import math
import dynamiqs as dq
from jaxtyping import Array
import numpy as np
import strawberryfields as sf
from jax import lax
import matplotlib.pyplot as plt
from IPython.display import display, Latex, Math
from functools import partial
import os

jax.clear_caches()


root2 = jnp.sqrt(2.0)

# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".95"
# dq.set_precision("double")

# Printing Utils
n_print = 25
jnp.set_printoptions(precision=3)


def mat_print(array, tol=3):
    matrix = ""
    for row in array:
        try:
            for number in row:
                matrix += f"{np.round(number, tol)}&"
        except TypeError:
            matrix += f"{row}&"
        matrix = matrix[:-1] + r"\\"
    display(Math(r"\begin{bmatrix}" + matrix + r"\end{bmatrix}"))


def lat_print(val):
    display(Latex(val))


def tprint(mat, tol=3):
    mat_print(mat[:n_print, :n_print], tol=tol)


@jax.jit
def dqtensor(*args):
    return dq.tensor(*args).to_jax()


@jax.jit
def dqdag(arg):
    return dq.dag(arg).to_jax()


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


# Finite GKP
Delta = 0.2
GKP_N = 100
GKP_L = 2.0 * jnp.sqrt(jnp.pi)
alpha = GKP_L * jnp.array([0, 1], dtype=jnp.complex64)
beta = GKP_L * jnp.array([-1, 0], dtype=complex)

n_hat = dqnumber(GKP_N)
a = dqdestroy(GKP_N)
a_dag = dqcreate(GKP_N)
x = (a + a_dag) / root2
p = -1j * (a - a_dag) / root2
p = p
x = x

sigma_x = dq.sigmax().to_jax()
sigma_y = dq.sigmay().to_jax()
sigma_z = dq.sigmaz().to_jax()


ket0 = dq.fock(2, 0)
ket1 = dq.fock(2, 1)
ketplus = (ket0 + ket1) / root2
ketminus = (ket0 - ket1) / root2

I2 = dqeye(2)
IN = dqeye(GKP_N)
II = dqtensor(IN, I2)
NI = dqtensor(n_hat, I2)
XI = dqtensor(x, I2)
PI = dqtensor(p, I2)
IZ = dqtensor(IN, sigma_z)
IX = dqtensor(IN, sigma_x)
IY = dqtensor(IN, sigma_y)
XZ = dqtensor(x, sigma_z)
PZ = dqtensor(p, sigma_z)

Ia = dqtensor(IN, dqdestroy(2))

l_j = jnp.array(
    [
        jnp.sqrt(alpha[0] ** 2 + beta[0] ** 2),
        jnp.sqrt(alpha[1] ** 2 + beta[1] ** 2),
    ]
)
q_j = jnp.array(
    [
        alpha[0] * x + beta[0] * p,
        alpha[1] * x + beta[1] * p,
    ]
)
q_j_perp = jnp.array(
    [
        alpha[0] * p - beta[0] * x,
        alpha[1] * p - beta[1] * x,
    ]
)
omega_12 = alpha[0] * beta[1] - beta[0] * alpha[1]
T_j_0 = jnp.array(
    [
        jla.expm(1j * q_j[0]),
        jla.expm(1j * q_j[1]),
    ]
)
X_0 = jla.expm(1j * q_j[0] / 2.0)
Z_0 = jla.expm(1j * q_j[1] / 2.0)
Y_0 = jla.expm(1j * (q_j[0] + q_j[1]) / 2.0)
x_j = jnp.array([q_j[0] / l_j[0], q_j[1] / l_j[1]])
x_j_perp = jnp.array([q_j_perp[0] / l_j[0], q_j_perp[1] / l_j[1]])

# if full_reload: - how to generate for new GKP_N
#     x_j_m = jnp.array(
#         [
#             sawtooth_fourier(x_j[0],2*pi/l_j[0]),
#             sawtooth_fourier(x_j[1],2*pi/l_j[1])
#         ]
#     )
#     jnp.save("fourier_saved.npy",np.asarray(x_j_m))
# else:
#     x_j_m = jnp.asarray(np.load("fourier_saved.npy"))
cwd = os.getcwd()
os.chdir(os.path.expanduser("~/Documents/GitHub/jiang-research/FiniteGKP/gkp_utils"))
# os.chdir(os.path.expanduser("/home/cjblake/repos/jiang-research/FiniteGKP/gkp_utils"))
x_j_m = jnp.asarray(np.load("fourier_saved.npy"))
os.chdir(cwd)

c_Delta = jnp.cosh(Delta**2)
s_Delta = jnp.sinh(Delta**2)
t_Delta = jnp.tanh(Delta**2)
m_j = 2 * jnp.pi / c_Delta / l_j
E_D = jla.expm(-(Delta**2) * n_hat)
E_D_plus = jla.expm(-(Delta**2) * (n_hat + IN))
E_D_minus = jla.expm(-(Delta**2) * (n_hat - IN))
E_D_inv = jla.inv(E_D)
c_n = 0.5 * (E_D_minus @ E_D_inv + E_D_plus @ E_D_inv)
s_n = 0.5 * (E_D_minus @ E_D_inv - E_D_plus @ E_D_inv)
T_j_E = jnp.array([E_D @ T_j_0[0] @ E_D_inv, E_D @ T_j_0[1] @ E_D_inv])
d_j_E = 1.0 / root2 * (x_j_m / jnp.sqrt(t_Delta) + 1j * x_j_perp * jnp.sqrt(t_Delta))
d_j_E_dag = np.array([jnp.conj(d_j.T) for d_j in d_j_E])
d_j_E_prod = np.array([d_j_E_dag[j] @ d_j_E[j] for j in [0, 1]])
X_E = jla.expm(0.5 * (1j * q_j[0] * c_Delta - q_j_perp[0] * s_Delta))
Z_E = jla.expm(0.5 * (1j * q_j[1] * c_Delta - q_j_perp[1] * s_Delta))
Y_E = 1j * Z_E @ X_E


def R_x(theta):
    return jla.expm(-1j * theta * sigma_x / 2)


def R_y(theta):
    return jla.expm(-1j * theta * sigma_y / 2)


def R_z(theta):
    return jla.expm(-1j * theta * sigma_z / 2)


def IR_x(theta):
    return dqtensor(IN, R_x(theta))


def IR_y(theta):
    return dqtensor(IN, R_y(theta))


def IR_z(theta):
    return dqtensor(IN, R_z(theta))


def com(ai, bi):
    return ai @ bi - bi @ ai


def anticom(ai, bi):
    return ai @ bi + bi @ ai


@jax.jit
def sinm(ai):
    return -0.5j * (jla.expm(1j * ai) - jla.expm(-1j * ai))


@jax.jit
def sawtooth_fourier(ai, mi, ni=30):
    # a is the matrix
    # ni is the fourier truncation
    # mi is the half-width of the pulse
    sum = jnp.zeros_like(ai, dtype="complex64")
    arg_a = ai * 2 * jnp.pi / mi
    for k in range(1, ni + 1):
        sum = sum + ((-1) ** k) / k * sinm(arg_a * k)
    return -mi / jnp.pi * sum


@jax.jit
def D(alpha_i: complex):
    return jla.expm(alpha_i * a_dag - jnp.conj(alpha_i) * a)


@jax.jit
def CD(beta_i: complex):
    return jla.expm(
        dqtensor(beta_i * a_dag - jnp.conj(beta_i) * a, sigma_z) / (2 * root2)
    )


@jax.jit
def S(xi):
    return jla.expm(jnp.conj(xi) * (a @ a) - xi * (a_dag @ a_dag))


# Hamiltonian & SBS
gamma = 1.0  # free parameter
gamma_j = jnp.array([gamma, gamma])
T = 10
epsilon_j = s_Delta * 4 * jnp.pi / l_j
theta_j = jnp.angle(alpha + 1j * beta)
Gamma_dt = t_Delta / 4 * c_Delta**2 * l_j**2
dt = Gamma_dt[0] / gamma
t_f = dt * T
b_k = (sigma_x + 1j * sigma_y) / 2  # typo in paper
b_dag_k = dqdag(b_k)
Nt = 100
H_idle = dqtensor(IN, I2)
H_E_n = jnp.sqrt(gamma) * (dqtensor(d_j_E[0], b_dag_k) + dqtensor(d_j_E_dag[0], b_k))
U_n = jla.expm(-1j * jnp.sqrt(dt) * H_E_n)
U_n_dag = dqdag(U_n)
Number_rq = dqtensor(dqnumber(GKP_N), I2)
a_rq = dqtensor(a, I2)

CD_A = jnp.array([CD(epsilon_j[i] * jnp.exp(1.0j * theta_j[i])) for i in [0, 1]])
CD_B = jnp.array([CD(-1.0j * (alpha[i] + 1.0j * beta[i]) * c_Delta) for i in [0, 1]])
CD_A_small = jnp.array(
    [CD(epsilon_j[i] * jnp.exp(1.0j * theta_j[i]) / 2.0) for i in [0, 1]]
)
U_sBs = jnp.array(
    [
        CD_A_small[i]
        @ (
            dqtensor(IN, dqdag(R_x(jnp.pi / 2.0)))
            @ (CD_B[i] @ (dqtensor(IN, R_x(jnp.pi / 2.0)) @ (CD_A_small[i])))
        )
        for i in [0, 1]
    ]
)

# generate states
prog_gkp_fock = sf.Program(1)
with prog_gkp_fock.context as quantum_context:
    sf.ops.GKP(state=[0, 0], epsilon=Delta**2) | quantum_context
eng = sf.Engine("fock", backend_options={"cutoff_dim": GKP_N, "hbar": 1})
logical_zero = jnp.array(eng.run(prog_gkp_fock).state.data).reshape((GKP_N, 1))

prog_gkp_fock2 = sf.Program(1)
with prog_gkp_fock2.context as quantum_context:
    sf.ops.GKP(state=[np.pi, 0], epsilon=Delta**2) | quantum_context
logical_one = jnp.array(eng.run(prog_gkp_fock2).state.data).reshape((GKP_N, 1))
U = jnp.hstack((logical_zero, logical_one))
U_dag = dqdag(U)
U_ident = U_dag @ U
U_proj = U @ U_dag


# Manifold Operations
def Pi(rho):
    return U_proj @ rho @ U_proj


def Pi_perp(rho):
    return rho - Pi(rho)


def Pidot_2(sigma_dot):
    return 0.5 * (
        sigma_x * dqtrace(sigma_x @ sigma_dot)
        + sigma_y * dqtrace(sigma_y @ sigma_dot)
        + sigma_z * dqtrace(sigma_z @ sigma_dot)
    )


def Pidot(rho_dot):
    return U @ Pidot_2(U_dag @ rho_dot @ U) @ U_dag


def Pidot_perp(rho_dot):
    return rho_dot - Pidot(rho_dot)


def sigma_proj(rho):
    return U_dag @ rho @ U


def psi_C(theta: float, phi: float):
    return (
        jnp.cos(theta / 2.0) * logical_zero
        + jnp.exp(1j * phi) * jnp.sin(theta / 2.0) * logical_one
    )


def rho_C(theta: float, phi: float):
    return dqtodm(psi_C(theta=theta, phi=phi))


def compute_bloch_vector(rho):
    sigma = sigma_proj(rho)
    return dqexpect([sigma_x, sigma_y, sigma_z], sigma)


def add_rho_to_bloch(rho, blocher):
    vec = compute_bloch_vector(rho)
    blocher.add_vectors(vec)


# Loss Channel
loss_gamma = 0.1


def make_loss_ops(loss_gamma_in, rank):
    return jnp.array(
        [
            (loss_gamma_in / (1 - loss_gamma_in)) ** (l / 2)
            / jnp.sqrt(math.factorial(l))
            * jnp.linalg.matrix_power(a, l)
            @ jla.expm(jnp.log(1 - loss_gamma_in) * n_hat / 2)
            for l in range(rank)
        ]
    )


# Kraus Operations
def verify_kraus_ops(ops, rtol=1e-6, truncate_final=False):
    k, n, _ = ops.shape
    output = jnp.zeros((n, n), dtype="complex64")
    for i in range(k):  # TODO jaxify
        output = output + dqdag(ops[i, :, :]) @ ops[i, :, :]
    if truncate_final:
        print(jnp.diagonal(output))
        return jnp.max(jnp.abs(jnp.eye(n - 1) - output[:-1, :-1]))
    else:
        return jnp.max(jnp.abs(jnp.eye(n) - output))


@jax.jit
def verify_kraus_ops_print(ops):
    contract = jax.jit(jax.vmap(lambda A: dqdag(A) @ A, in_axes=0))
    return jnp.sum(contract(ops), axis=0)


@jax.jit
def apply_kraus_map(ops: Array, rho: Array):
    def apply_single(op: Array):
        return op @ rho @ dqdag(op)

    partials = jax.jit(jax.vmap(apply_single, in_axes=0))(ops)
    rho_out = jnp.sum(partials, axis=0)
    return rho_out / dqtrace(rho_out)


@jax.jit
def apply_kraus_map_nonorm(ops: Array, rho: Array):
    def apply_single(op: Array):
        return op @ rho @ dqdag(op)

    partials = jax.jit(jax.vmap(apply_single, in_axes=0))(ops)
    return jnp.sum(partials, axis=0)


@jax.jit
def apply_kraus_map_n(ops: Array, rho: Array, n: int):
    def body_loop(i, rho_loop):
        return apply_kraus_map(ops, rho_loop)

    return jax.lax.fori_loop(0, n, body_loop, rho)


@jax.jit
def compose_channel_kraus(ch1, ch2):
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


@partial(jax.jit, static_argnums=2)
def generate_cycling_gif(ops: Array, rho_0: Array, cycles: int):
    evo = jnp.zeros((cycles, GKP_N, GKP_N), dtype="complex64")
    evo = evo.at[0, :, :].set(rho_0)

    def body_fun(i, mat):
        return mat.at[i, :, :].set(apply_kraus_map(ops, mat[i - 1, :, :]))

    return jax.lax.fori_loop(1, cycles, body_fun, evo)


def super_compose(ops: Array, times):
    evo = jnp.expand_dims(jnp.eye(ops.shape[2], dtype="complex64"), axis=0)
    for i in range(times):
        evo = compose_channel_kraus(ops, evo)
    return evo


def vec_colwise(A: jnp.ndarray) -> jnp.ndarray:
    return A.flatten(order="F")


# Choi Operations
@jax.jit
def kraus_to_choi(kraus_ops: jnp.ndarray) -> jnp.ndarray:
    k, n, _ = kraus_ops.shape

    def body_fun(i, current_sum):
        K_i = kraus_ops[i]
        vK_i = vec_colwise(K_i)
        return current_sum + jnp.outer(vK_i, jnp.conjugate(vK_i))

    choi_init = jnp.zeros((n * n, n * n), dtype=kraus_ops.dtype)
    choi = jax.lax.fori_loop(0, k, body_fun, choi_init)
    return choi


def unvec_colwise(x: jnp.ndarray, n: int) -> jnp.ndarray:
    return x.reshape((n, n), order="F")


# @jax.jit
def choi_to_kraus(choi: jnp.ndarray, rtol=1e-10) -> jnp.ndarray:
    n2 = choi.shape[0]
    n = int(jnp.sqrt(n2))

    eigvals, eigvecs = jnp.linalg.eigh(choi)

    max_eig = jnp.amax(eigvals)
    mask = eigvals > (rtol * max_eig)

    eigvals_nonzero = eigvals[mask]
    eigvecs_nonzero = eigvecs[:, mask]  # shape (n^2, k')

    def build_one_kraus(i):
        lam = eigvals_nonzero[i]
        uvec = eigvecs_nonzero[:, i]
        return jnp.sqrt(lam) * unvec_colwise(uvec, n)

    k = eigvals_nonzero.shape[0]
    kraus_ops = jax.vmap(build_one_kraus)(jnp.arange(k))
    return kraus_ops


# Characteristic Functions
def make_transpose_for_loss(loss_ops_in):
    P = U_proj
    loss_P = apply_kraus_map_nonorm(loss_ops_in, P)
    loss_P_eigs, loss_P_vecs = jnp.linalg.eigh(loss_P)

    def supp_invsqrt(arr):
        return jnp.where(arr != 0, arr**-0.5, arr)

    loss_P_eigs2 = supp_invsqrt(jnp.round(loss_P_eigs, decimals=3))
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


@jax.jit
def generate_characteristic_function(operators, mesh_uy, mesh_ux, mesh_vy, mesh_vx):
    # operators \in C^(kraus_op_count,dim_H,dim_H) # independent of Kraus representation
    # meshes \in R^{dim_mesh_i}
    ux, uy = jnp.meshgrid(mesh_ux, mesh_uy)
    U = ux - 1j * uy
    vx, vy = jnp.meshgrid(mesh_vx, mesh_vy)
    V = vx - 1j * vy
    c_uv = jnp.zeros(
        (mesh_ux.shape[0], mesh_uy.shape[0], mesh_vx.shape[0], mesh_vy.shape[0]),
        dtype="complex64",
    )

    def evaluate_operation(E_i, i, j, k, l):
        return dqtrace(
            E_i @ dqdag(dqdisplace(GKP_N, U[i, j]))
        )  # this has to regenerate displacement operators because I literally do not have enough memory to precompute them all..

    full_map_at_ijklE = jax.vmap(
        evaluate_operation, in_axes=(0, None, None, None, None)
    )
    full_map_at_ijkl = jax.jit(partial(full_map_at_ijklE, operators))

    @jax.jit
    def body_l(l, arr_l, i, j, k):
        return arr_l.at[i, j, k, l].set(jnp.sum(full_map_at_ijkl(i, j, k, l)))

    @jax.jit
    def body_k(k, arr_k, i, j):
        return lax.fori_loop(
            0, mesh_vy.shape[0], lambda l, arr_l: body_l(l, arr_l, i, j, k), arr_k
        )

    @jax.jit
    def body_j(j, arr_j, i):
        return lax.fori_loop(
            0, mesh_vx.shape[0], lambda k, arr_k: body_k(k, arr_k, i, j), arr_j
        )

    @jax.jit
    def body_i(i, arr_i):
        return lax.fori_loop(
            0, mesh_uy.shape[0], lambda j, arr_j: body_j(j, arr_j, i), arr_i
        )

    return lax.fori_loop(0, mesh_ux.shape[0], body_i, c_uv)


def map_char_func(operators, fname, title, meshes):
    mesh_str = f"_{meshes[0].shape[0]}_{meshes[1].shape[0]}_{meshes[2].shape[0]}_{meshes[3].shape[0]}"
    if not os.path.exists(fname + mesh_str + ".npy"):
        c_uv_out = generate_characteristic_function(
            operators, meshes[1], meshes[0], meshes[3], meshes[2]
        )
        jnp.save(fname + mesh_str + ".npy", np.asarray(c_uv_out))
    else:
        c_uv_out = jnp.load(fname + mesh_str + ".npy")
    plt.imshow(
        jnp.real(c_uv_out[:, :, 0, 0]),
        extent=(meshes[0][0], meshes[0][-1], meshes[1][0], meshes[1][-1]),
    )
    plt.title(title)
    plt.xlabel(r"$u_x$")
    plt.ylabel(r"$u_y$")
    cbar = plt.colorbar()
    plt.savefig(fname + "_real" + mesh_str + ".png")
    plt.show()

    plt.clf()
    plt.imshow(
        jnp.abs(c_uv_out[:, :, 0, 0]),
        extent=(meshes[0][0], meshes[0][-1], meshes[1][0], meshes[1][-1]),
    )
    plt.title("(mag) " + title)
    plt.xlabel(r"$u_x$")
    plt.ylabel(r"$u_y$")
    cbar = plt.colorbar()
    plt.savefig(fname + "_abs" + mesh_str + ".png")
    plt.show()

    return c_uv_out


mesh_ux = jnp.linspace(-5, 5, 160)
mesh_uy = jnp.linspace(-5, 5, 160)
mesh_vx = jnp.linspace(0, 0, 1)
mesh_vy = jnp.linspace(0, 0, 1)
meshes_default_small = [mesh_ux, mesh_uy, mesh_vx, mesh_vy]


def net_qubit_action(params):
    nqa = jnp.eye(2)
    for i in range(params.shape[0]):
        nqa = R_theta_phi(params[i, 2], params[i, 1]) @ nqa
    return nqa


# Gates From "Real-Time QEC Beyond Breakeven" p20
@jax.jit
def R_theta_phi(theta, phi):
    return jla.expm(
        -1j * (theta / 2) * (sigma_x * jnp.cos(phi) + sigma_y * jnp.sin(phi))
    )


@jax.jit
def ECD(beta: complex):
    return dqtensor(IN, sigma_x) @ CD(beta * root2)


@jax.jit
def circuit_layer(vals):
    # beta,phi,theta = vals
    return ECD(vals[0]) @ dqtensor(IN, R_theta_phi(vals[2], vals[1]))


@jax.jit
def compose_ECD_layers(params: Array):
    if params.shape[1] != 3:
        raise TypeError
    circuit = circuit_layer(params[0, :])

    def body_mult(i, circ):
        return circuit_layer(params[i, :]) @ circ

    return jax.lax.fori_loop(1, params.shape[0], body_mult, circuit)


# Lev-Arcady Sellem Gates
def R_n_theta(theta):  # S196
    return jnp.diag(jnp.exp(jnp.arange(0, GKP_N, dtype="complex64") * 1j * theta))


las_eta = 2 * jnp.sqrt(jnp.pi)
las_Delta = 0.4
las_epsilon = las_eta * jnp.sinh(las_Delta)
las_dissipators = jnp.zeros((4, GKP_N, GKP_N), dtype="complex64")
for i in range(4):
    rot_i = R_n_theta(i * jnp.pi / 2)
    las_dissipators = las_dissipators.at[i, :, :].set(
        jnp.exp(-las_eta * las_epsilon / 2)
        * rot_i
        @ dqdisplace(GKP_N, 1j * las_eta / root2)
        @ (IN - las_epsilon * p)
        @ dqdag(rot_i)
        - IN
    )  # eq3
las_M_kraus = jnp.zeros((5, GKP_N, GKP_N), dtype="complex64")
las_dt = 0.1
las_M_kraus = las_M_kraus.at[0, :, :].set(
    IN - 0.5 * las_dt * verify_kraus_ops_print(las_dissipators)
)
for i in range(1, 5):
    las_M_kraus = las_M_kraus.at[i, :, :].set(jnp.sqrt(las_dt) * las_dissipators[i - 1])
MM = verify_kraus_ops_print(las_M_kraus)
MMeigs, MMvecs = jla.eigh(MM)
invsqrtR = MMvecs @ jnp.diag(MMeigs**-0.5) @ dqdag(MMvecs)
las_N_kraus = jnp.array([las_M_kraus[i, :, :] @ invsqrtR for i in range(5)])
