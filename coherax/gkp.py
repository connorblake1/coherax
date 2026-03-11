"""GKP code state construction in the coherent basis.

Constructs finite-energy Gottesman-Kitaev-Preskill (GKP) code states
as :class:`~coherax.states.CoherentKet` superpositions on square or
rectangular lattices.
"""

from __future__ import annotations

import dynamiqs as dq
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np
from jaxtyping import Array
from scipy.signal import fftconvolve

from coherax.states import CoherentKet


def gkp_coherent_dm(
    mu: int,
    N_trunc: int,
    Delta: float,
    lattice: str = "rect",
    lam: float = jnp.sqrt(2.0),
    N_trunc_y: int | None = None,
) -> CoherentKet:
    r"""Construct a finite-energy GKP code word as a coherent superposition.

    Builds the logical :math:`|{\mu}_L\rangle` (:math:`\mu \in \{0, 1\}`)
    on the specified lattice with Gaussian envelope parameter ``Delta``.

    Parameters
    ----------
    mu : int
        Logical index (0 or 1).
    N_trunc : int
        Lattice truncation in the :math:`\alpha`-direction.
    Delta : float
        Envelope squeezing parameter.
    lattice : str
        ``"square"`` or ``"rect"`` (rectangular with aspect ratio ``lam``).
    lam : float
        Aspect ratio for the rectangular lattice (ignored for ``"square"``).
    N_trunc_y : int or None
        Lattice truncation in the :math:`\beta`-direction. Defaults to
        ``N_trunc`` if not given.

    Returns
    -------
    CoherentKet
        Normalized coherent-state superposition representing the code word.

    References
    ----------
    Grimsmo & Puri, "Quantum Error Correction with the
    Gottesman-Kitaev-Preskill Code," PRX Quantum (2021).
    """
    if N_trunc_y is None:
        N_trunc_y = N_trunc
    if lattice == "square":
        GKP_alpha = jnp.sqrt(jnp.pi / 2)
        GKP_beta = 1.0j * jnp.sqrt(jnp.pi / 2)
    elif lattice == "rect":
        GKP_alpha = lam * jnp.sqrt(jnp.pi / 2)
        GKP_beta = 1.0j * jnp.sqrt(jnp.pi / 2) / lam
    else:
        raise ValueError(f"Unknown lattice type {lattice!r}; use 'square' or 'rect'.")

    cs: list[Array] = []
    ds: list[Array] = []
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


# ---------------------------------------------------------------------------
# GKP diagnostics
# ---------------------------------------------------------------------------


def stabilizer_expectations(
    rho: Array, Ncav: int | None = None
) -> tuple[float, float, float, float]:
    r"""Compute GKP stabilizer expectations.

    Returns ``(|<S1>|, |<S2>|, Delta_eff_1, Delta_eff_2)`` where
    :math:`S_1 = D(\sqrt{2\pi})` tests x-periodicity and
    :math:`S_2 = D(i\sqrt{2\pi})` tests p-periodicity.

    Parameters
    ----------
    rho : Array, shape ``(N, N)``
        Density matrix.
    Ncav : int or None
        Cavity truncation (defaults to ``rho.shape[0]``).

    Returns
    -------
    tuple of four floats
        ``(|<S1>|, |<S2>|, Delta_eff_1, Delta_eff_2)``
    """
    if Ncav is None:
        Ncav = rho.shape[0]
    a = dq.destroy(Ncav).to_jax()
    a_dag = dq.create(Ncav).to_jax()
    alpha_1 = np.sqrt(2 * np.pi)
    alpha_2 = 1j * np.sqrt(2 * np.pi)
    S1 = jla.expm(alpha_1 * a_dag - np.conj(alpha_1) * a)
    S2 = jla.expm(alpha_2 * a_dag - np.conj(alpha_2) * a)
    abs_S1 = float(jnp.abs(jnp.trace(S1 @ rho[:Ncav, :Ncav])))
    abs_S2 = float(jnp.abs(jnp.trace(S2 @ rho[:Ncav, :Ncav])))

    def delta_eff(abs_S: float) -> float:
        if 0 < abs_S < 1:
            return float(np.sqrt(-2 * np.log(abs_S) / np.pi))
        return 0.0 if abs_S >= 1 else float("inf")

    return abs_S1, abs_S2, delta_eff(abs_S1), delta_eff(abs_S2)


def fock_wavefunctions(N: int, x_grid: np.ndarray) -> np.ndarray:
    r"""Hermite wavefunctions :math:`\psi_n(x)` for :math:`n=0,\dots,N-1` via recurrence.

    Parameters
    ----------
    N : int
        Number of Fock states.
    x_grid : ndarray, shape ``(M,)``
        Position grid.

    Returns
    -------
    ndarray, shape ``(N, M)``
    """
    psi = np.zeros((N, len(x_grid)))
    psi[0] = np.pi ** (-0.25) * np.exp(-(x_grid**2) / 2)
    if N > 1:
        psi[1] = np.sqrt(2) * x_grid * psi[0]
    for n in range(2, N):
        psi[n] = np.sqrt(2 / n) * x_grid * psi[n - 1] - np.sqrt((n - 1) / n) * psi[n - 2]
    return psi


def x_marginal(rho_np: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    r"""Position-space marginal :math:`p(x) = \langle x|\rho|x\rangle`.

    Parameters
    ----------
    rho_np : ndarray, shape ``(N, N)``
        Density matrix (NumPy array).
    x_grid : ndarray, shape ``(M,)``
        Position grid.

    Returns
    -------
    ndarray, shape ``(M,)``
    """
    N = rho_np.shape[0]
    psi = fock_wavefunctions(N, x_grid)
    return np.real(np.einsum("mi,mn,ni->i", psi, rho_np, psi))


def gkp_x_error_rate(
    rho: Array,
    sigma_noise: float,
    d_lattice: float,
    mu: int = 0,
    N_fock: int = 80,
) -> float:
    r"""Logical error probability in the x-quadrature under Gaussian displacement noise.

    Parameters
    ----------
    rho : Array, shape ``(N, N)``
        Density matrix.
    sigma_noise : float
        Standard deviation of Gaussian displacement noise.
    d_lattice : float
        GKP lattice period.
    mu : int
        Logical index (0 or 1).
    N_fock : int
        Fock truncation for the marginal computation.

    Returns
    -------
    float
        Logical error probability.
    """
    rho_np = np.array(rho[:N_fock, :N_fock])
    rho_np = rho_np / np.trace(rho_np)
    x_grid = np.linspace(-15, 15, 6001)
    dx = x_grid[1] - x_grid[0]
    px = x_marginal(rho_np, x_grid)
    px = np.maximum(px, 0)
    px /= px.sum() * dx
    if sigma_noise > 0:
        kernel = np.exp(-(x_grid**2) / (2 * sigma_noise**2))
        kernel /= kernel.sum() * dx
        px = fftconvolve(px, kernel, mode="same") * dx
    shifted = x_grid - mu * d_lattice / 2
    modular = shifted - d_lattice * np.round(shifted / d_lattice)
    correct = np.abs(modular) < d_lattice / 4
    return 1 - np.sum(px[correct]) * dx
