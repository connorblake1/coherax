import pytest
import itertools
import dynamiqs as dq
import jax.numpy as jnp
from jaxtyping import Array
import jax
from coherax.characteristic_jax_utils import (
    R_z,
    R_y,
    qubit_rotation,
    dqcoherent_dm,
    dqdag,
    dqdisplace,
    dqptrace,
    dqtensor,
    dqtodm,
    ecd_rotation_2x2,
    ECD,
    CD,
    IN,
    channel_from_b,
    krauscompose,
    TraceoutLayer,
    g,
    CoherentKet,
    params_to_charfunc,
    analytic_fidelity,
    circuit_layer,
    compose_ECD_layers,
    circuit_params_to_2channel,
    batch_circuit_params_to_2channel,
    compact_channel_to_exec_channel,
)

GKP_N = 100
tol = 1e-5


@pytest.mark.parametrize(
    "cs, ds",
    [
        (jnp.array([1.0, 1.0]), jnp.array([2.0, -2.0])),
        (jnp.array([1.0]), jnp.array([4.0])),
        (
            jnp.array([1, 1, 1, 1]),
            jnp.array([2.0 + 2.0j, 2.0 - 2.0j, -2.0 + 2.0j, -2 - 2.0j]),
        ),
    ],
)
def test_coherent_dm(cs, ds):
    c_rho_cat = CoherentKet(
        cs=cs,
        ds=ds,
    )
    assert jnp.abs(c_rho_cat(0) - 1.0) < tol


@pytest.mark.parametrize(
    "circuit_parameters, N_l, cs, ds",
    [
        (jnp.array([[2.0, 0.0, 0.0, 0.0]]), 4, jnp.array([1.0]), jnp.array([1.0])),
        (
            jnp.array(
                [[2.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0]]
            ),  # flipped bc of ECD
            8,
            jnp.array([1.0]),
            jnp.array([2.0]),
        ),
    ],
)
def test_params_to_coherent(circuit_parameters: Array, N_l: int, cs: Array, ds: Array):
    cdm = CoherentKet(
        cs=cs,
        ds=ds,
    )
    X, Y = jnp.meshgrid(jnp.linspace(-4, 4, 20), jnp.linspace(-4, 4, 20))
    Z = X.ravel() + 1.0j * Y.ravel()
    coherent_values = jax.vmap(cdm)(Z)
    constructed_fn = params_to_charfunc(circuit_parameters=circuit_parameters, N_l=N_l)
    constructed_values = jax.vmap(constructed_fn)(Z)
    assert jnp.max(jnp.abs(coherent_values - constructed_values)) < tol


@pytest.mark.parametrize(
    "cs, ds",
    [
        (jnp.array([1.0, 1.0]), jnp.array([2.0, -2.0])),
        (jnp.array([1.0]), jnp.array([4.0])),
        (
            jnp.array([1, 1, 1, 1]),
            jnp.array([2.0 + 2.0j, 2.0 - 2.0j, -2.0 + 2.0j, -2 - 2.0j]),
        ),
    ],
)
def test_analytic_fidelity(cs: Array, ds: Array):
    c_coherent_dm = CoherentKet(cs=cs, ds=ds)
    alpha_cohere = jnp.expand_dims(c_coherent_dm.cs, 0)
    beta_cohere = jnp.expand_dims(c_coherent_dm.ds, 0)
    assert (
        jnp.abs(
            1.0
            - analytic_fidelity(alpha_cohere, alpha_cohere, beta_cohere, beta_cohere)
        )
        < tol
    )


@pytest.mark.parametrize(
    "u", [2.0 - 2.0j, 0.5 + 0.1j, -1.2 + 0.7j, 0.0 + 1.0j, -0.3 - 0.8j]
)
def test_cd_channel_matches_coherent_mixture(u):
    psi_plus = (dq.fock(2, 0) + dq.fock(2, 1)) / jnp.sqrt(2.0)
    psi_plus_dm = dqtodm(psi_plus)
    rho0 = dq.fock_dm(GKP_N, 0)
    rho_total = dqtensor(rho0, psi_plus_dm)
    U_CD_u = CD(u)
    rho_evolved = dqptrace(U_CD_u @ rho_total @ dqdag(U_CD_u), 0, (GKP_N, 2))
    rho_analytic = (dqcoherent_dm(GKP_N, u) + dqcoherent_dm(GKP_N, -u)) / 2.0
    assert jnp.max(jnp.abs(rho_analytic - rho_evolved)) < tol


@pytest.mark.parametrize(
    "angles",
    [
        [0.345, -0.644, 3.14],
        [1.454, -5.734, 0.0],
        [0.0, jnp.pi / 3, jnp.pi / 6],
        [jnp.pi, -jnp.pi / 7, jnp.pi / 7],
    ],
)
def test_rotations(angles):
    phi = angles[0]
    theta = angles[1]
    gamma = angles[2]
    U_composite = qubit_rotation(phi=phi, theta=theta, gamma=gamma)
    U_decomposed = R_z(phi) @ R_y(theta) @ R_z(gamma)
    print(U_composite, U_decomposed)
    assert jnp.max(jnp.abs(U_composite - U_decomposed)) < tol


dummy_parameters = [
    [jnp.pi, jnp.pi / 2, 3.1, 0.5 - 2j],
    [1.5, -5.7, 0.0, -2.0 + 1.0j],
    [0.0, jnp.pi / 3, jnp.pi / 6, 0.34 - 2.7j],
    [jnp.pi, -jnp.pi / 7, jnp.pi / 7, 0.1],
]


@pytest.mark.parametrize("angles_d", dummy_parameters)
def test_ecd_rotations(angles_d):
    phi = angles_d[0]
    theta = angles_d[1]
    gamma = angles_d[2]
    d = angles_d[3]
    U_composite = ecd_rotation_2x2(phi=phi, theta=theta, gamma=gamma)
    Dp = dqdisplace(GKP_N, d / 2)
    Dm = dqdisplace(GKP_N, -d / 2)
    U_layer = dqtensor(
        Dm, jnp.array([[U_composite[0, 0], U_composite[0, 1]], [0.0, 0.0]])
    ) + dqtensor(Dp, jnp.array([[0.0, 0.0], [U_composite[1, 0], U_composite[1, 1]]]))
    U_from_def = circuit_layer(jnp.array([d, phi, theta, gamma], jnp.complex64))
    difference_mat = U_layer - U_from_def
    assert jnp.max(jnp.abs(difference_mat)) < tol


@pytest.mark.parametrize("angles_d", dummy_parameters)
def test_single_layer(angles_d):
    phi = jnp.real(angles_d[0])
    theta = jnp.real(angles_d[1])
    gamma = jnp.real(angles_d[2])
    d = jnp.array(angles_d[3]).astype(jnp.complex64)
    circuit_parameters = jnp.array([[d, phi, theta, gamma]], jnp.complex64)
    U_analytic = circuit_layer(jnp.array([d, phi, theta, gamma], jnp.complex64))
    K_0_analytic = (
        dqtensor(IN, dqdag(dq.fock(2, 0))) @ U_analytic @ dqtensor(IN, dq.fock(2, 0))
    )
    K_1_analytic = (
        dqtensor(IN, dqdag(dq.fock(2, 1))) @ U_analytic @ dqtensor(IN, dq.fock(2, 0))
    )

    layer_alphas, layer_betas = g(circuit_params=circuit_parameters, N_l=2)
    layer_channel = channel_from_b(layer_alphas, layer_betas)

    assert (jnp.max(jnp.abs(K_0_analytic - layer_channel[0, :, :])) < tol) and (
        jnp.max(jnp.abs(K_1_analytic - layer_channel[1, :, :])) < tol
    )


@pytest.mark.parametrize(
    "angles_d_1, angles_d_2", list(itertools.product(dummy_parameters, repeat=2))
)
def test_layer_sequential_2(angles_d_1, angles_d_2):
    phi_1 = jnp.real(angles_d_1[0])
    phi_2 = jnp.real(angles_d_2[0])
    theta_1 = jnp.real(angles_d_1[1])
    theta_2 = jnp.real(angles_d_2[1])
    gamma_1 = jnp.real(angles_d_1[2])
    gamma_2 = jnp.real(angles_d_2[2])
    d_1 = jnp.array(angles_d_1[3]).astype(jnp.complex64)
    d_2 = jnp.array(angles_d_2[3]).astype(jnp.complex64)
    circuit_parameters = jnp.array(
        [
            [
                d_1,
                phi_1,
                theta_1,
                gamma_1,
            ],
            [
                d_2,
                phi_2,
                theta_2,
                gamma_2,
            ],
        ],
        jnp.complex64,
    )
    print(circuit_parameters)
    U_analytic = compose_ECD_layers(circuit_parameters)
    K_0_analytic = (
        dqtensor(IN, dqdag(dq.fock(2, 0))) @ U_analytic @ dqtensor(IN, dq.fock(2, 0))
    )
    K_1_analytic = (
        dqtensor(IN, dqdag(dq.fock(2, 1))) @ U_analytic @ dqtensor(IN, dq.fock(2, 0))
    )

    layer_alphas, layer_betas = g(circuit_params=circuit_parameters, N_l=4)
    print("alphas")
    print(layer_alphas)
    print("betas")
    print(layer_betas)
    composed_channel = channel_from_b(alphas=layer_alphas, betas=layer_betas)
    diff0 = K_0_analytic - composed_channel[0, :, :]
    diff1 = K_1_analytic - composed_channel[1, :, :]
    print(jnp.argmax(jnp.abs(diff0)), jnp.argmax(jnp.abs(diff1)))
    assert (
        jnp.max(jnp.abs(diff0[:60, :60])) < tol
        and jnp.max(jnp.abs(diff1[:60, :60])) < tol
    )


dummy_parameters_small = [
    [jnp.pi, jnp.pi / 2, 3.1, 0.1 - 0.1j],
    [1.5, -5.7, 0.0, -0.20],
    [jnp.pi, -jnp.pi / 7, jnp.pi / 7, 1.0],
]


@pytest.mark.parametrize(
    "angles_d_1, angles_d_2, angles_d_3, angles_d_4",
    list(itertools.product(dummy_parameters_small, repeat=4)),
)
def test_layer_sequential_4(angles_d_1, angles_d_2, angles_d_3, angles_d_4):
    phi_1 = jnp.real(angles_d_1[0])
    phi_2 = jnp.real(angles_d_2[0])
    phi_3 = jnp.real(angles_d_3[0])
    phi_4 = jnp.real(angles_d_3[0])
    theta_1 = jnp.real(angles_d_1[1])
    theta_2 = jnp.real(angles_d_2[1])
    theta_3 = jnp.real(angles_d_3[1])
    theta_4 = jnp.real(angles_d_4[1])
    gamma_1 = jnp.real(angles_d_1[2])
    gamma_2 = jnp.real(angles_d_2[2])
    gamma_3 = jnp.real(angles_d_3[2])
    gamma_4 = jnp.real(angles_d_4[2])
    d_1 = jnp.array(angles_d_1[3]).astype(jnp.complex64)
    d_2 = jnp.array(angles_d_2[3]).astype(jnp.complex64)
    d_3 = jnp.array(angles_d_3[3]).astype(jnp.complex64)
    d_4 = jnp.array(angles_d_4[3]).astype(jnp.complex64)

    circuit_parameters = jnp.array(
        [
            [
                d_1,
                phi_1,
                theta_1,
                gamma_1,
            ],
            [
                d_2,
                phi_2,
                theta_2,
                gamma_2,
            ],
            [
                d_3,
                phi_3,
                theta_3,
                gamma_3,
            ],
            [
                d_4,
                phi_4,
                theta_4,
                gamma_4,
            ],
        ],
        jnp.complex64,
    )
    print(circuit_parameters)
    U_analytic = compose_ECD_layers(circuit_parameters)
    K_0_analytic = (
        dqtensor(IN, dqdag(dq.fock(2, 0))) @ U_analytic @ dqtensor(IN, dq.fock(2, 0))
    )
    K_1_analytic = (
        dqtensor(IN, dqdag(dq.fock(2, 1))) @ U_analytic @ dqtensor(IN, dq.fock(2, 0))
    )

    layer_alphas, layer_betas = g(circuit_params=circuit_parameters, N_l=16)
    composed_channel = channel_from_b(alphas=layer_alphas, betas=layer_betas)
    diff0 = K_0_analytic - composed_channel[0, :, :]
    diff1 = K_1_analytic - composed_channel[1, :, :]
    assert (
        jnp.max(jnp.abs(diff0[:60, :60])) < tol
        and jnp.max(jnp.abs(diff1[:60, :60])) < tol
    )


@pytest.mark.parametrize(
    "angles_d_1, angles_d_2", list(itertools.product(dummy_parameters, repeat=2))
)
def test_composer(angles_d_1, angles_d_2):
    phi_1 = jnp.real(angles_d_1[0])
    phi_2 = jnp.real(angles_d_2[0])
    theta_1 = jnp.real(angles_d_1[1])
    theta_2 = jnp.real(angles_d_2[1])
    gamma_1 = jnp.real(angles_d_1[2])
    gamma_2 = jnp.real(angles_d_2[2])
    d_1 = angles_d_1[3]
    d_2 = angles_d_2[3]
    U_analytic = (
        ECD(beta=d_2)
        @ dqtensor(IN, qubit_rotation(phi=phi_2, theta=theta_2, gamma=gamma_2))
        @ ECD(beta=d_1)
        @ dqtensor(IN, qubit_rotation(phi=phi_1, theta=theta_1, gamma=gamma_1))
    )
    circuit_parameters = jnp.array(
        [
            [d_1, phi_1, theta_1, gamma_1],
            [d_2, phi_2, theta_2, gamma_2],
        ],
        jnp.complex64,
    )
    print(circuit_parameters)
    U_composed = compose_ECD_layers(circuit_parameters)
    assert jnp.max(jnp.abs(U_analytic - U_composed)) < tol


@pytest.mark.parametrize(
    "angles_d_1, angles_d_2", list(itertools.product(dummy_parameters, repeat=2))
)
def test_layer_traceout_2(angles_d_1, angles_d_2):
    phi_1 = jnp.real(angles_d_1[0])
    phi_2 = jnp.real(angles_d_2[0])
    theta_1 = jnp.real(angles_d_1[1])
    theta_2 = jnp.real(angles_d_2[1])
    gamma_1 = jnp.real(angles_d_1[2])
    gamma_2 = jnp.real(angles_d_2[2])
    d_1 = jnp.array(angles_d_1[3]).astype(jnp.complex64)
    d_2 = jnp.array(angles_d_2[3]).astype(jnp.complex64)
    circuit_1 = jnp.array([d_1, phi_1, theta_1, gamma_1])
    circuit_2 = jnp.array([d_2, phi_2, theta_2, gamma_2])

    l_a = TraceoutLayer.from_single_param(circuit_layer=circuit_1, N_l=2)
    l_b = TraceoutLayer.from_single_param(circuit_layer=circuit_2, N_l=2)
    l_alpha_composite, l_beta_composite = krauscompose(l_a=l_b, l_b=l_a)

    channel = channel_from_b(l_alpha_composite, l_beta_composite)
    params_1 = jnp.array([[d_1, phi_1, theta_1, gamma_1]], jnp.complex64)
    params_2 = jnp.array([[d_2, phi_2, theta_2, gamma_2]], jnp.complex64)
    kraus_1 = circuit_params_to_2channel(params_1)
    kraus_2 = circuit_params_to_2channel(params_2)
    kraus_00 = kraus_2[0] @ kraus_1[0]
    kraus_01 = kraus_2[0] @ kraus_1[1]
    kraus_10 = kraus_2[1] @ kraus_1[0]
    kraus_11 = kraus_2[1] @ kraus_1[1]
    diff0 = kraus_00 - channel[0, :, :]
    diff1 = kraus_01 - channel[1, :, :]
    diff2 = kraus_10 - channel[2, :, :]
    diff3 = kraus_11 - channel[3, :, :]
    max_N = 60
    assert (
        jnp.max(jnp.abs(diff0[:max_N, :max_N])) < tol
        and jnp.max(jnp.abs(diff1[:max_N, :max_N])) < tol
        and jnp.max(jnp.abs(diff2[:max_N, :max_N])) < tol
        and jnp.max(jnp.abs(diff3[:max_N, :max_N])) < tol
    )


@pytest.mark.parametrize(
    "angles_d_1, angles_d_2", list(itertools.product(dummy_parameters, repeat=2))
)
def test_channel_batch(angles_d_1, angles_d_2):
    phi_1 = jnp.real(angles_d_1[0])
    phi_2 = jnp.real(angles_d_2[0])
    theta_1 = jnp.real(angles_d_1[1])
    theta_2 = jnp.real(angles_d_2[1])
    gamma_1 = jnp.real(angles_d_1[2])
    gamma_2 = jnp.real(angles_d_2[2])
    d_1 = angles_d_1[3]
    d_2 = angles_d_2[3]

    total_params = jnp.array(
        [[[d_1, phi_1, theta_1, gamma_1]], [[d_2, phi_2, theta_2, gamma_2]]]
    )
    print(total_params.shape)
    compact_channel = batch_circuit_params_to_2channel(total_params)
    print("compact", compact_channel.shape)
    channel = compact_channel_to_exec_channel(compact_channel, 2)
    print(channel.shape)
    params_1 = jnp.array([[d_1, phi_1, theta_1, gamma_1]], jnp.complex64)
    params_2 = jnp.array([[d_2, phi_2, theta_2, gamma_2]], jnp.complex64)
    kraus_1 = circuit_params_to_2channel(params_1)
    kraus_2 = circuit_params_to_2channel(params_2)

    kraus_00 = kraus_2[0] @ kraus_1[0]
    kraus_01 = kraus_2[1] @ kraus_1[0]
    kraus_02 = kraus_2[0] @ kraus_1[1]
    kraus_03 = kraus_2[1] @ kraus_1[1]
    diff0 = kraus_00 - channel[0, :, :]
    diff1 = kraus_01 - channel[1, :, :]
    diff2 = kraus_02 - channel[2, :, :]
    diff3 = kraus_03 - channel[3, :, :]
    max_N = 60
    assert (
        jnp.max(jnp.abs(diff0[:max_N, :max_N])) < tol
        and jnp.max(jnp.abs(diff1[:max_N, :max_N])) < tol
        and jnp.max(jnp.abs(diff2[:max_N, :max_N])) < tol
        and jnp.max(jnp.abs(diff3[:max_N, :max_N])) < tol
    )


# def test_super_g(angles_d_1, angles_d_2): # TODO
#     pass
