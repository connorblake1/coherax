from typing import List
from jaxtyping import Array
import jax.numpy as jnp
import jax.scipy.linalg as jla
import jax
import numpy as np
import dynamiqs as dq
from .utils import GKP_N
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

displacement_epsilon = 0.005


class CompositeDisplacement:
    ops: List

    def __init__(self, l):
        self.ops = l

    def __mul__(self, other):
        if isinstance(other, Displacement):
            return CompositeDisplacement([op * other for op in self.ops])
        elif isinstance(other, CompositeDisplacement):
            new_list = [
                op_self * op_other for op_self in self.ops for op_other in other.ops
            ]
            return CompositeDisplacement(
                [op for op in new_list if jnp.abs(op.coeff) > displacement_epsilon]
            )
        else:
            raise TypeError

    def __add__(self, other):
        if isinstance(other, Displacement):
            return CompositeDisplacement[self.ops.append(other)]
        elif isinstance(other, CompositeDisplacement):
            new_list = self.ops + other.ops
            return CompositeDisplacement(
                [op for op in new_list if jnp.abs(op.coeff) > displacement_epsilon]
            )
        else:
            raise TypeError

    def __repr__(self):
        out = "Composite["
        for item in self.ops:
            out = out + item.__repr__()
        out = out + "]"
        return out


class Displacement:
    coeff: complex
    alpha: complex

    def __init__(self, a, c=1.0):
        self.alpha = a
        self.coeff = c

    def __rmul__(self, other):  # the one on the right is the caller
        if isinstance(other, Displacement):
            return other * self
        if isinstance(other, CompositeDisplacement):
            return CompositeDisplacement([op * self for op in other.ops])
        return self._scalar_multiply(other)

    def _scalar_multiply(self, other):
        if np.isscalar(other) or (isinstance(other, jax.Array) and other.ndim == 0):
            return Displacement(a=self.alpha, c=self.coeff * other)
        else:
            raise TypeError

    def __mul__(self, other):  # left is the caller
        if isinstance(other, Displacement):
            return (
                self.coeff
                * other.coeff
                * jnp.exp(
                    (
                        self.alpha * jnp.conj(other.alpha)
                        - jnp.conj(self.alpha) * other.alpha
                    )
                    / 2
                )
                * Displacement(self.alpha + other.alpha)
            )
        elif isinstance(other, CompositeDisplacement):
            return CompositeDisplacement([self * op for op in other.ops])
        else:
            return self._scalar_multiply(other)

    def __add__(self, other):
        return CompositeDisplacement([self, other])

    def __repr__(self):
        return f"(a={self.alpha:.3f},c={self.coeff:.3f})"


def symbolic_circuit_layer(params: Array):
    # bottom of p36 in Jiang Notes, unitaries only (pre traceout)
    beta, theta, phi = params[0], params[2], params[1]
    return np.array(
        [
            [
                CompositeDisplacement(
                    [
                        Displacement(-beta / 2)
                        * (
                            2 * jnp.cos(phi / 2) * jnp.sin(phi / 2) * jnp.sin(theta / 2)
                            - 1j * jnp.sin(theta / 2) * jnp.cos(phi)
                        )
                    ]
                ),
                CompositeDisplacement([Displacement(-beta / 2) * jnp.cos(theta / 2)]),
            ],
            [
                CompositeDisplacement([Displacement(beta / 2) * jnp.cos(theta / 2)]),
                CompositeDisplacement(
                    [
                        Displacement(beta / 2)
                        * (
                            -2
                            * jnp.cos(phi / 2)
                            * jnp.sin(phi / 2)
                            * jnp.sin(theta / 2)
                            - 1j * jnp.sin(theta / 2) * jnp.cos(phi)
                        )
                    ]
                ),
            ],
        ]
    )


def symbolic_circuit(params: Array) -> np.ndarray:
    # symbolic product of circuits
    circuit = symbolic_circuit_layer(params[0, :])
    for i in range(1, params.shape[0]):
        circuit = symbolic_circuit_layer(params[i, :]) @ circuit
    return circuit


def symbolic_to_numeric(mat: np.ndarray) -> Array:
    # takes symbolic circuit, assumes preparation in ket0 (ie only the first column), and then turns into Kraus operator set of size 2 (single traceout)
    output = jnp.zeros((2, GKP_N, GKP_N), dtype="complex64")
    for i in [0, 1]:
        if isinstance(mat[i, 0], Displacement):
            output = output.at[i, :, :].set(
                mat[i, 0].coeff * dq.displace(GKP_N, mat[i, 0].alpha)
            )
        elif isinstance(mat[i, 0], CompositeDisplacement):
            for op in mat[i, 0].ops:
                output = output.at[i, :, :].add(op.coeff * dq.displace(GKP_N, op.alpha))
    return output


def display_circuit(circ: np.ndarray, intermediates=True):
    x_all = np.array([])
    y_all = np.array([])
    cmag_all = np.array([])
    cphase_all = np.array([])
    for i in [0, 1]:
        for j in [0, 1]:
            if isinstance(circ[i, j], Displacement):
                xs = np.array([circ[i, j].alpha.real])
                ys = np.array([circ[i, j].alpha.imag])
                cms = np.array([np.abs(circ[i, j].coeff)])
                cps = np.array([np.angle(circ[i, j].coeff)])
            else:
                xs = np.array([op.alpha.real for op in circ[i, j].ops])
                ys = np.array([op.alpha.imag for op in circ[i, j].ops])
                cms = np.array([np.abs(op.coeff) for op in circ[i, j].ops])
                cps = np.array([np.angle(op.coeff) for op in circ[i, j].ops])
            x_all = np.concatenate((x_all, xs))
            y_all = np.concatenate((y_all, ys))
            cmag_all = np.concatenate((cmag_all, cms))
            cphase_all = np.concatenate((cphase_all, cps))
            max_mag = np.max(cms)
            if intermediates:
                fig, ax = plt.subplots()
                cmap = plt.cm.viridis
                norm = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)
                scatter = ax.scatter(
                    xs, ys, c=cps, s=cms / max_mag * 40, cmap=cmap, norm=norm
                )
                cbar = plt.colorbar(scatter, ax=ax, label="Phase (radians)")
                cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
                cbar.set_ticklabels(
                    [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
                )
                ax.set_xlabel("Real")
                ax.set_ylabel("Imag")
                ax.set_title(f"Characteristic Function Dirac Weights ({i},{j})")
                plt.show()
    plt.clf()
    max_mag = np.max(cmag_all)
    fig, ax = plt.subplots()
    cmap = plt.cm.viridis
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


# Numeric-Symbolic Conversions
from .utils import ket0, ket1, IN


def Rot(theta, phi):
    return (
        jnp.cos(theta / 2) * dq.eye(2)
        - 1j * jnp.sin(theta / 2) * jnp.cos(phi) * dq.sigmax()
        - 2j * jnp.cos(phi / 2) * jnp.sin(phi / 2) * jnp.sin(theta / 2) * dq.sigmay()
    )


def ECD2(beta):
    return dq.tensor(IN, dq.sigmax()) @ jla.expm(
        dq.tensor(
            dq.create(GKP_N) * beta - jnp.conj(beta) * dq.destroy(GKP_N), dq.sigmaz()
        )
        / 2
    )


def CD22(beta):
    return dq.tensor(dq.displace(GKP_N, beta), dq.fock_dm(2, 0)) + dq.tensor(
        dq.displace(GKP_N, -beta), dq.fock_dm(2, 1)
    )


def analytic_circuit_layer(beta, theta, phi):
    return (
        dq.tensor(dq.displace(GKP_N, -beta / 2), dq.fock_dm(2, 0))
        * (
            2 * jnp.cos(phi / 2) * jnp.sin(phi / 2) * jnp.sin(theta / 2)
            - 1j * jnp.sin(theta / 2) * jnp.cos(phi)
        )
        + dq.tensor(dq.displace(GKP_N, beta / 2), dq.fock_dm(2, 1))
        * (
            -2 * jnp.cos(phi / 2) * jnp.sin(phi / 2) * jnp.sin(theta / 2)
            - 1j * jnp.sin(theta / 2) * jnp.cos(phi)
        )
        + dq.tensor(dq.displace(GKP_N, beta / 2), ket1 @ dq.dag(ket0))
        * jnp.cos(theta / 2)
        + dq.tensor(dq.displace(GKP_N, -beta / 2), ket0 @ dq.dag(ket1))
        * jnp.cos(theta / 2)
    )
