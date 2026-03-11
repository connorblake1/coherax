"""coherax — Coherent-basis optimization for bosonic quantum codes.

Public API is organized into submodules:

- :mod:`coherax.operators` — Quantum operators, dynamiqs wrappers, constants
- :mod:`coherax.states` — CoherentKet, CoherentDM, BosonicSubspace
- :mod:`coherax.circuits` — CD+R circuit construction, TraceoutLayer, ``g()``
- :mod:`coherax.fidelity` — Analytic fidelity computations
- :mod:`coherax.gkp` — GKP code state generators
"""

from coherax.operators import (
    GKP_N,
    IN,
    I2,
    aOmegab,
    a_op,
    a_dag_op,
    apply_kraus_map,
    apply_kraus_map_n,
    apply_kraus_map_nonorm,
    coherent_overlap,
    compose_channel_kraus,
    dag,
    dqcoherent,
    dqcoherent_dm,
    dqcreate,
    dqdag,
    dqdestroy,
    dqdisplace,
    dqeye,
    dqexpect,
    dqfock_dm,
    dqnumber,
    dqptrace,
    dqsqueeze,
    dqtensor,
    dqtodm,
    dqtrace,
    e_n1iaOmegab,
    invsqrtm,
    ket0,
    ket1,
    make_pureloss_fock,
    make_thermalloss_fock,
    make_transpose_for_pureloss,
    n_hat,
    p_quad,
    root2,
    sigma_x,
    sigma_y,
    sigma_z,
    sparse_eigh,
    sparse_tensor_eigh,
    von_neumann_entropy,
    x_quad,
)

from coherax.states import BosonicSubspace, CoherentDM, CoherentKet

from coherax.circuits import (
    CD,
    ECD,
    R_x,
    R_y,
    R_z,
    TraceoutLayer,
    W,
    channel_from_b,
    circuit_layer,
    circuit_params_to_2channel,
    circuit_params_to_time,
    compose_ECD_layers,
    ecd_rotation_2x2,
    g,
    gate_timer,
    qubit_rotation,
    super_g,
    traceout_unitary,
)

from coherax.fidelity import (
    analytic_fidelity,
    analytic_fidelity_i,
    analytic_fidelity_transfer,
    analytic_fidelity_transfer_i,
    analytic_fidelity_wrapper,
    analytic_pureloss_recovery_fidelity_thetaphi,
    analytic_pureloss_recovery_fidelity_thetaphi_iab,
)

from coherax.gkp import (
    fock_wavefunctions,
    gkp_coherent_dm,
    gkp_x_error_rate,
    stabilizer_expectations,
    x_marginal,
)

from coherax.info import (
    coherent_info_from_coherent_kets,
    coherent_info_from_kets,
    coherent_info_thermal_fock,
)

__all__ = [
    # operators
    "GKP_N", "IN", "I2", "root2",
    "sigma_x", "sigma_y", "sigma_z",
    "n_hat", "a_op", "a_dag_op", "x_quad", "p_quad", "ket0", "ket1",
    "aOmegab", "e_n1iaOmegab", "coherent_overlap",
    "dag", "invsqrtm", "sparse_eigh", "sparse_tensor_eigh",
    "dqtensor", "dqdag", "dqeye", "dqnumber", "dqdestroy", "dqcreate",
    "dqtrace", "dqdisplace", "dqsqueeze", "dqfock_dm", "dqcoherent_dm",
    "dqcoherent", "dqptrace", "dqexpect", "dqtodm",
    "apply_kraus_map", "apply_kraus_map_n", "apply_kraus_map_nonorm",
    "compose_channel_kraus", "make_pureloss_fock", "make_thermalloss_fock",
    "make_transpose_for_pureloss", "von_neumann_entropy",
    # states
    "CoherentKet", "CoherentDM", "BosonicSubspace",
    # circuits
    "W", "CD", "ECD", "R_x", "R_y", "R_z",
    "qubit_rotation", "ecd_rotation_2x2",
    "circuit_layer", "compose_ECD_layers", "traceout_unitary",
    "circuit_params_to_2channel",
    "TraceoutLayer", "g", "channel_from_b", "super_g",
    "gate_timer", "circuit_params_to_time",
    # fidelity
    "analytic_fidelity", "analytic_fidelity_i",
    "analytic_fidelity_transfer", "analytic_fidelity_transfer_i",
    "analytic_fidelity_wrapper",
    "analytic_pureloss_recovery_fidelity_thetaphi",
    "analytic_pureloss_recovery_fidelity_thetaphi_iab",
    # gkp
    "gkp_coherent_dm", "stabilizer_expectations", "fock_wavefunctions",
    "x_marginal", "gkp_x_error_rate",
    # info
    "coherent_info_from_kets", "coherent_info_from_coherent_kets",
    "coherent_info_thermal_fock",
]
