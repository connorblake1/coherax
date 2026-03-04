"""GKP code state construction in the coherent basis.

Constructs finite-energy Gottesman-Kitaev-Preskill (GKP) code states
as :class:`~coherax.states.CoherentKet` superpositions on square or
rectangular lattices.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

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
