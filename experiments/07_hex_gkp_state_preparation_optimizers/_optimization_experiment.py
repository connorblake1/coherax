"""End-to-end ECD optimization for a finite-energy hex-GKP target.

This file is copied verbatim into the self-contained experiment notebook by
``_build_notebook.py``.  The notebook, not this helper file, is the canonical
executable record.
"""

from __future__ import annotations

import json
import math
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import dynamiqs as dq
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax


# %% Explicit benchmark contract

EXPERIMENT_DIRECTORY = Path.cwd()
FIGURE_DIRECTORY = EXPERIMENT_DIRECTORY / "figs"
SEQUENCE_DIRECTORY = EXPERIMENT_DIRECTORY / "sequences"
FIGURE_DIRECTORY.mkdir(exist_ok=True)
SEQUENCE_DIRECTORY.mkdir(exist_ok=True)

LAYER_COUNTS = (4, 8, 12, 16, 20)
METHOD_NAMES = (
    "full_coherent",
    "direct_fock",
    "dense_numerical_fock",
    "squeezed_packets",
    "continuous_anchors",
)
METHOD_LABELS = {
    "full_coherent": "full coherent paths",
    "direct_fock": "analytic Fock recurrence",
    "dense_numerical_fock": "dense numerical Fock",
    "squeezed_packets": "squeezed packets",
    "continuous_anchors": "continuous anchors",
}
METHOD_COLORS = {
    "full_coherent": "#2667a9",
    "direct_fock": "#d1495b",
    "dense_numerical_fock": "#7b2cbf",
    "squeezed_packets": "#2a9d8f",
    "continuous_anchors": "#e09f3e",
}

TARGET_DELTA = 0.2
TARGET_LOGICAL_INDEX = 0
TARGET_HEX_SHELL_RADIUS = 14
TARGET_CONVERGENCE_REFERENCE_RADIUS = 20
RESTART_COUNT = int(os.environ.get("COHERAX_EXP07_RESTARTS", "3"))
OUTER_STEPS = int(os.environ.get("COHERAX_EXP07_STEPS", "10000"))
OUTER_LEARNING_RATE = float(os.environ.get("COHERAX_EXP07_LR", "0.008"))
FOCK_OPTIMIZER_CUTOFF = int(os.environ.get("COHERAX_EXP07_FOCK_CUTOFF", "48"))
PACKET_FOCK_CUTOFF = int(os.environ.get("COHERAX_EXP07_PACKET_CUTOFF", "48"))
PACKETS_PER_QUBIT_BRANCH = int(os.environ.get("COHERAX_EXP07_PACKETS_PER_BRANCH", "32"))
CONTINUOUS_CENTER_STEPS = int(os.environ.get("COHERAX_EXP07_CENTER_STEPS", "12"))
SQUEEZE_STEPS = int(os.environ.get("COHERAX_EXP07_SQUEEZE_STEPS", "16"))
VALIDATION_FOCK_CUTOFF = 256
WIGNER_FOCK_CUTOFF = 96
PARAMETER_ABSOLUTE_LIMIT = 5.0

INITIAL_COMPLEX_VARIANCE = 0.5
INITIAL_COMPONENT_STANDARD_DEVIATION = math.sqrt(INITIAL_COMPLEX_VARIANCE / 2.0)
INITIALIZATION_SEED_BASE = 2026080600

COHERAX_CHI_RADIANS_PER_SECOND = 2.0 * math.pi * 50e3
COHERAX_DRIVE_SCALE = 20.0
COHERAX_MINIMUM_DISPLACEMENT_SECONDS = 48e-9
COHERAX_ANCILLA_SECONDS = 24e-9

EICKBUSCH_CHI_RADIANS_PER_SECOND = 2.0 * math.pi * 33e3
EICKBUSCH_DRIVE_SCALE = 30.0
EICKBUSCH_QUBIT_PULSE_SECONDS = 24e-9
EICKBUSCH_DISPLACEMENT_PULSE_SECONDS = 44e-9
EICKBUSCH_LAYER_FLOOR_SECONDS = (
    2.0 * EICKBUSCH_QUBIT_PULSE_SECONDS
    + 4.0 * EICKBUSCH_DISPLACEMENT_PULSE_SECONDS
)


# %% Circuit formulas and the exact coherent-path score

def build_hex_gkp_target_numpy(
    shell_radius: int,
    delta: float = TARGET_DELTA,
    logical_index: int = TARGET_LOGICAL_INDEX,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the normalized finite hex-GKP coherent-state sum.

    The hexagonal lattice vectors are

        alpha = sqrt(pi / sqrt(3)),
        beta = exp(2*pi*i/3) alpha.

    A retained site has integer coordinates ``(m, l)`` satisfying
    ``max(|m|, |l|, |m-l|) <= shell_radius`` and ``m mod 2 = logical_index``.
    Writing ``k = (m-logical_index)/2``, its center and raw coefficient are

        d = m alpha + l beta,
        c = exp[-i*pi*(k*l + l*logical_index/2) - delta**2*|d|**2].

    The returned coefficients are normalized with the exact coherent Gram
    matrix, not with a Fock truncation.
    """
    if shell_radius < 1:
        raise ValueError("shell_radius must be at least one")
    if logical_index not in (0, 1):
        raise ValueError("logical_index must be zero or one")
    lattice_alpha = math.sqrt(math.pi / math.sqrt(3.0))
    lattice_beta = np.exp(2.0j * math.pi / 3.0) * lattice_alpha
    coefficients = []
    centers = []
    for first_index in range(-shell_radius, shell_radius + 1):
        for second_index in range(-shell_radius, shell_radius + 1):
            if (
                max(
                    abs(first_index),
                    abs(second_index),
                    abs(first_index - second_index),
                )
                > shell_radius
                or first_index % 2 != logical_index
            ):
                continue
            logical_first_index = (first_index - logical_index) // 2
            center = first_index * lattice_alpha + second_index * lattice_beta
            coefficient = np.exp(
                -1.0j
                * math.pi
                * (
                    logical_first_index * second_index
                    + second_index * logical_index / 2.0
                )
                - delta**2 * abs(center) ** 2
            )
            coefficients.append(coefficient)
            centers.append(center)
    coefficients_array = np.asarray(coefficients, dtype=np.complex128)
    centers_array = np.asarray(centers, dtype=np.complex128)
    gram = np.exp(
        -0.5
        * np.abs(centers_array[:, None] - centers_array[None, :]) ** 2
        + 1.0j
        * (
            centers_array[:, None].real * centers_array[None, :].imag
            - centers_array[:, None].imag * centers_array[None, :].real
        )
    )
    norm_squared = float(
        np.real(
            np.sum(
                np.conj(coefficients_array)[:, None]
                * coefficients_array[None, :]
                * gram
            )
        )
    )
    return coefficients_array / math.sqrt(norm_squared), centers_array


def coherent_superposition_fock_numpy(
    coefficients: np.ndarray,
    centers: np.ndarray,
    cutoff: int,
) -> np.ndarray:
    """Convert a coherent superposition to its first ``cutoff`` Fock entries."""
    current_columns = np.exp(-0.5 * np.abs(centers) ** 2).astype(np.complex128)
    fock_state = np.empty(cutoff, dtype=np.complex128)
    fock_state[0] = np.sum(coefficients * current_columns)
    for fock_number in range(1, cutoff):
        current_columns = current_columns * centers / math.sqrt(fock_number)
        fock_state[fock_number] = np.sum(coefficients * current_columns)
    return fock_state


def coherent_superposition_overlap_numpy(
    first_coefficients: np.ndarray,
    first_centers: np.ndarray,
    second_coefficients: np.ndarray,
    second_centers: np.ndarray,
    second_chunk_size: int = 16384,
) -> complex:
    """Exact coherent overlap, chunked to bound temporary memory."""
    overlap = 0.0 + 0.0j
    for chunk_start in range(0, second_centers.size, second_chunk_size):
        chunk_stop = min(chunk_start + second_chunk_size, second_centers.size)
        chunk_centers = second_centers[chunk_start:chunk_stop]
        chunk_coefficients = second_coefficients[chunk_start:chunk_stop]
        gram_chunk = np.exp(
            -0.5 * np.abs(first_centers[:, None] - chunk_centers[None, :]) ** 2
            + 1.0j
            * (
                first_centers[:, None].real * chunk_centers[None, :].imag
                - first_centers[:, None].imag * chunk_centers[None, :].real
            )
        )
        overlap += np.sum(
            np.conj(first_coefficients)[:, None]
            * chunk_coefficients[None, :]
            * gram_chunk
        )
    return complex(overlap)


TARGET_COEFFICIENTS_NUMPY, TARGET_CENTERS_NUMPY = build_hex_gkp_target_numpy(
    TARGET_HEX_SHELL_RADIUS
)
TARGET_REFERENCE_COEFFICIENTS_NUMPY, TARGET_REFERENCE_CENTERS_NUMPY = (
    build_hex_gkp_target_numpy(TARGET_CONVERGENCE_REFERENCE_RADIUS)
)
TARGET_SHELL_REFERENCE_FIDELITY = abs(
    coherent_superposition_overlap_numpy(
        TARGET_COEFFICIENTS_NUMPY,
        TARGET_CENTERS_NUMPY,
        TARGET_REFERENCE_COEFFICIENTS_NUMPY,
        TARGET_REFERENCE_CENTERS_NUMPY,
    )
) ** 2
TARGET_COEFFICIENTS_JAX = jnp.asarray(TARGET_COEFFICIENTS_NUMPY)
TARGET_CENTERS_JAX = jnp.asarray(TARGET_CENTERS_NUMPY)
TARGET_FOCK_OPTIMIZER_NUMPY = coherent_superposition_fock_numpy(
    TARGET_COEFFICIENTS_NUMPY,
    TARGET_CENTERS_NUMPY,
    FOCK_OPTIMIZER_CUTOFF,
)
TARGET_FOCK_PACKET_NUMPY = coherent_superposition_fock_numpy(
    TARGET_COEFFICIENTS_NUMPY,
    TARGET_CENTERS_NUMPY,
    PACKET_FOCK_CUTOFF,
)
TARGET_FOCK_VALIDATION_NUMPY = coherent_superposition_fock_numpy(
    TARGET_COEFFICIENTS_NUMPY,
    TARGET_CENTERS_NUMPY,
    VALIDATION_FOCK_CUTOFF,
)
TARGET_FOCK_WIGNER_NUMPY = coherent_superposition_fock_numpy(
    TARGET_COEFFICIENTS_NUMPY,
    TARGET_CENTERS_NUMPY,
    WIGNER_FOCK_CUTOFF,
)
TARGET_BARGMANN_COEFFICIENTS_NUMPY = (
    np.conj(TARGET_FOCK_OPTIMIZER_NUMPY)
    * np.exp(
        -0.5
        * np.asarray(
            [math.lgamma(fock_number + 1.0) for fock_number in range(FOCK_OPTIMIZER_CUTOFF)]
        )
    )
)
TARGET_BARGMANN_COEFFICIENTS_JAX = jnp.asarray(
    TARGET_BARGMANN_COEFFICIENTS_NUMPY[::2]
)
TARGET_BARGMANN_DERIVATIVE_COEFFICIENTS_JAX = jnp.asarray(
    np.arange(1, (FOCK_OPTIMIZER_CUTOFF + 1) // 2)
    * TARGET_BARGMANN_COEFFICIENTS_NUMPY[2::2]
)
TARGET_FOCK_OPTIMIZER_NORM = float(
    np.vdot(TARGET_FOCK_OPTIMIZER_NUMPY, TARGET_FOCK_OPTIMIZER_NUMPY).real
)
TARGET_MEAN_PHOTON_VALIDATION = float(
    np.sum(
        np.arange(VALIDATION_FOCK_CUTOFF)
        * np.abs(TARGET_FOCK_VALIDATION_NUMPY) ** 2
    )
    / np.vdot(TARGET_FOCK_VALIDATION_NUMPY, TARGET_FOCK_VALIDATION_NUMPY).real
)


def parameters_from_real(real_parameters: jnp.ndarray) -> jnp.ndarray:
    """Map [Re(beta), Im(beta), phi, theta] to [beta, phi, theta, gamma=0]."""
    complex_parameters = jnp.zeros(
        (real_parameters.shape[0], 4), dtype=jnp.complex128
    )
    complex_parameters = complex_parameters.at[:, 0].set(
        real_parameters[:, 0] + 1j * real_parameters[:, 1]
    )
    complex_parameters = complex_parameters.at[:, 1].set(real_parameters[:, 2])
    complex_parameters = complex_parameters.at[:, 2].set(real_parameters[:, 3])
    return complex_parameters


def real_from_parameters(complex_parameters: np.ndarray) -> np.ndarray:
    """Inverse of :func:`parameters_from_real` for saved ECD sequences."""
    complex_parameters = np.asarray(complex_parameters)
    return np.column_stack(
        (
            complex_parameters[:, 0].real,
            complex_parameters[:, 0].imag,
            complex_parameters[:, 1].real,
            complex_parameters[:, 2].real,
        )
    )


def ecd_rotation_jax(phi: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Qubit rotation with the ECD row swap already included."""
    prefactor = jnp.exp(-0.5j * phi)
    return prefactor * jnp.asarray(
        [
            [
                jnp.sin(theta / 2.0) * jnp.exp(1j * phi),
                jnp.cos(theta / 2.0) * jnp.exp(1j * phi),
            ],
            [jnp.cos(theta / 2.0), -jnp.sin(theta / 2.0)],
        ],
        dtype=jnp.complex128,
    )


def symplectic_jax(first: jnp.ndarray, second: jnp.ndarray) -> jnp.ndarray:
    return jnp.real(first) * jnp.imag(second) - jnp.imag(first) * jnp.real(second)


def expand_coherent_paths_jax(
    real_parameters: jnp.ndarray,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    """Expand the exact circuit into its 2**n coherent paths."""
    parameters = parameters_from_real(real_parameters)
    first_beta = parameters[0, 0]
    first_rotation = ecd_rotation_jax(
        jnp.real(parameters[0, 1]), jnp.real(parameters[0, 2])
    )
    coefficients = (
        jnp.asarray([first_rotation[0, 0]]),
        jnp.asarray([first_rotation[1, 0]]),
    )
    centers = (
        jnp.asarray([-first_beta / 2.0]),
        jnp.asarray([first_beta / 2.0]),
    )
    for layer_index in range(1, real_parameters.shape[0]):
        beta = parameters[layer_index, 0]
        rotation = ecd_rotation_jax(
            jnp.real(parameters[layer_index, 1]),
            jnp.real(parameters[layer_index, 2]),
        )
        output_coefficients = []
        output_centers = []
        for output_qubit, shift in enumerate((-beta / 2.0, beta / 2.0)):
            branch_coefficients = []
            branch_centers = []
            for input_qubit in range(2):
                phase = jnp.exp(
                    -1j * symplectic_jax(shift, centers[input_qubit])
                )
                branch_coefficients.append(
                    rotation[output_qubit, input_qubit]
                    * coefficients[input_qubit]
                    * phase
                )
                branch_centers.append(centers[input_qubit] + shift)
            output_coefficients.append(jnp.concatenate(branch_coefficients))
            output_centers.append(jnp.concatenate(branch_centers))
        coefficients = (output_coefficients[0], output_coefficients[1])
        centers = (output_centers[0], output_centers[1])
    return coefficients, centers


def target_overlap_from_paths_jax(
    coefficients: tuple[jnp.ndarray, jnp.ndarray],
    centers: tuple[jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Exact fidelity against the normalized finite hex-GKP coherent sum."""
    amplitudes = []
    for qubit_branch in range(2):
        branch_centers = centers[qubit_branch]
        target_bra = jnp.zeros(branch_centers.shape, dtype=jnp.complex128)
        # Process a small number of lattice sites at a time. This is the exact
        # finite sum, but avoids materializing all target-site/path pairs.
        target_chunk_size = 8
        for chunk_start in range(0, TARGET_CENTERS_NUMPY.size, target_chunk_size):
            chunk_stop = min(
                chunk_start + target_chunk_size,
                TARGET_CENTERS_NUMPY.size,
            )
            target_centers = TARGET_CENTERS_JAX[chunk_start:chunk_stop]
            target_coefficients = TARGET_COEFFICIENTS_JAX[chunk_start:chunk_stop]

            @jax.checkpoint
            def add_target_chunk(current_target_bra, current_centers):
                coherent_overlaps = jnp.exp(
                    -0.5
                    * jnp.abs(
                        target_centers[:, None] - current_centers[None, :]
                    )
                    ** 2
                    + 1.0j
                    * symplectic_jax(
                        target_centers[:, None], current_centers[None, :]
                    )
                )
                return current_target_bra + jnp.sum(
                    jnp.conj(target_coefficients)[:, None] * coherent_overlaps,
                    axis=0,
                )

            target_bra = add_target_chunk(target_bra, branch_centers)
        amplitudes.append(jnp.sum(coefficients[qubit_branch] * target_bra))
    return jnp.real(jnp.abs(amplitudes[0]) ** 2 + jnp.abs(amplitudes[1]) ** 2)


@jax.custom_jvp
def projected_target_bra_jax(centers: jnp.ndarray) -> jnp.ndarray:
    r"""Return <target_48|center> from its Bargmann polynomial."""
    squared_centers = centers**2
    polynomial = jnp.polyval(
        TARGET_BARGMANN_COEFFICIENTS_JAX[::-1],
        squared_centers,
    )
    return jnp.exp(-0.5 * jnp.abs(centers) ** 2) * polynomial


@projected_target_bra_jax.defjvp
def projected_target_bra_jvp(primals, tangents):
    """Memory-bounded analytic directional derivative of the polynomial."""
    (centers,) = primals
    (center_tangents,) = tangents
    squared_centers = centers**2
    polynomial = jnp.polyval(
        TARGET_BARGMANN_COEFFICIENTS_JAX[::-1],
        squared_centers,
    )
    squared_derivative_polynomial = jnp.polyval(
        TARGET_BARGMANN_DERIVATIVE_COEFFICIENTS_JAX[::-1],
        squared_centers,
    )
    derivative_polynomial = 2.0 * centers * squared_derivative_polynomial
    gaussian = jnp.exp(-0.5 * jnp.abs(centers) ** 2)
    primal_output = gaussian * polynomial
    gaussian_direction = -0.5 * (
        jnp.conj(centers) * center_tangents
        + centers * jnp.conj(center_tangents)
    )
    tangent_output = gaussian * (
        derivative_polynomial * center_tangents
        + polynomial * gaussian_direction
    )
    return primal_output, tangent_output


def projected_target_fidelity_from_paths_jax(
    coefficients: tuple[jnp.ndarray, jnp.ndarray],
    centers: tuple[jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Training fidelity with the normalized 48-level target projection."""
    amplitudes = [
        jnp.sum(
            coefficients[qubit_branch]
            * projected_target_bra_jax(centers[qubit_branch])
        )
        for qubit_branch in range(2)
    ]
    return jnp.real(
        jnp.abs(amplitudes[0]) ** 2 + jnp.abs(amplitudes[1]) ** 2
    ) / TARGET_FOCK_OPTIMIZER_NORM


def full_coherent_fidelity_jax(real_parameters: jnp.ndarray) -> jnp.ndarray:
    """Exact coherent paths scored against the normalized 48-level target."""
    coefficients, centers = expand_coherent_paths_jax(real_parameters)
    return projected_target_fidelity_from_paths_jax(coefficients, centers)


def exact_coherent_score_numpy(real_parameters: np.ndarray) -> float:
    """Independent float64 host score used to rank every restart."""
    real_parameters = np.asarray(real_parameters, dtype=np.float64)
    coefficients = [np.asarray([1.0 + 0.0j]), np.empty(0, dtype=np.complex128)]
    centers = [np.asarray([0.0 + 0.0j]), np.empty(0, dtype=np.complex128)]
    for beta_real, beta_imag, phi, theta in real_parameters:
        beta = complex(beta_real, beta_imag)
        prefactor = np.exp(-0.5j * phi)
        rotation = prefactor * np.asarray(
            [
                [
                    np.sin(theta / 2.0) * np.exp(1j * phi),
                    np.cos(theta / 2.0) * np.exp(1j * phi),
                ],
                [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
            ],
            dtype=np.complex128,
        )
        output_coefficients = []
        output_centers = []
        for output_qubit, shift in enumerate((-beta / 2.0, beta / 2.0)):
            coefficient_parts = []
            center_parts = []
            for input_qubit in range(2):
                if coefficients[input_qubit].size == 0:
                    continue
                symplectic = (
                    shift.real * centers[input_qubit].imag
                    - shift.imag * centers[input_qubit].real
                )
                coefficient_parts.append(
                    rotation[output_qubit, input_qubit]
                    * coefficients[input_qubit]
                    * np.exp(-1j * symplectic)
                )
                center_parts.append(centers[input_qubit] + shift)
            output_coefficients.append(np.concatenate(coefficient_parts))
            output_centers.append(np.concatenate(center_parts))
        coefficients, centers = output_coefficients, output_centers
    amplitudes = []
    for qubit_branch in range(2):
        amplitudes.append(
            coherent_superposition_overlap_numpy(
                TARGET_COEFFICIENTS_NUMPY,
                TARGET_CENTERS_NUMPY,
                coefficients[qubit_branch],
                centers[qubit_branch],
            )
        )
    fidelity = float(sum(abs(amplitude) ** 2 for amplitude in amplitudes))
    return float(np.clip(fidelity, 0.0, 1.0))


# %% Two Fock simulators

def coherent_columns_jax(
    centers: jnp.ndarray, cutoff: int
) -> jnp.ndarray:
    numbers = jnp.arange(cutoff, dtype=jnp.float64)[:, None]
    inverse_sqrt_factorial = jnp.exp(
        -0.5 * jax.scipy.special.gammaln(numbers + 1.0)
    )
    return (
        jnp.exp(-0.5 * jnp.abs(centers)[None, :] ** 2)
        * centers[None, :] ** numbers
        * inverse_sqrt_factorial
    )


def displacement_matrix_jax(displacement: jnp.ndarray, cutoff: int) -> jnp.ndarray:
    """Projected infinite-dimensional <m|D(displacement)|n> matrix."""
    first_column = coherent_columns_jax(
        jnp.asarray([displacement]), cutoff
    )[:, 0]
    raising_factors = jnp.sqrt(jnp.arange(1, cutoff, dtype=jnp.float64))

    def recurrence(previous_column, input_number):
        current_column = -jnp.conj(displacement) * previous_column
        current_column = current_column.at[1:].add(
            raising_factors * previous_column[:-1]
        )
        current_column = current_column / jnp.sqrt(input_number)
        return current_column, current_column

    _, remaining_columns = jax.lax.scan(
        recurrence,
        first_column,
        jnp.arange(1, cutoff, dtype=jnp.float64),
    )
    return jnp.column_stack((first_column, remaining_columns.T))


def fock_step_jax(
    state: jnp.ndarray, layer_parameters: jnp.ndarray, cutoff: int
) -> jnp.ndarray:
    beta = layer_parameters[0]
    rotation = ecd_rotation_jax(
        jnp.real(layer_parameters[1]), jnp.real(layer_parameters[2])
    )
    rotated_state = rotation @ state
    negative_displacement = displacement_matrix_jax(-beta / 2.0, cutoff)
    positive_displacement = displacement_matrix_jax(beta / 2.0, cutoff)
    return jnp.stack(
        (
            negative_displacement @ rotated_state[0],
            positive_displacement @ rotated_state[1],
        )
    )


def direct_fock_state_jax(
    real_parameters: jnp.ndarray, cutoff: int
) -> jnp.ndarray:
    state = jnp.zeros((2, cutoff), dtype=jnp.complex128)
    state = state.at[0, 0].set(1.0)
    parameters = parameters_from_real(real_parameters)
    for layer_index in range(real_parameters.shape[0]):
        state = fock_step_jax(state, parameters[layer_index], cutoff)
    return state


def normalized_target_fidelity_from_fock_jax(
    state: jnp.ndarray,
    target_fock: jnp.ndarray,
) -> jnp.ndarray:
    """Fidelity with the normalized target projection in this Fock cutoff."""
    target_amplitudes = jnp.sum(
        jnp.conj(target_fock)[None, :] * state,
        axis=1,
    )
    state_norm = jnp.real(jnp.vdot(state, state))
    target_norm = jnp.real(jnp.vdot(target_fock, target_fock))
    return jnp.real(jnp.sum(jnp.abs(target_amplitudes) ** 2)) / jnp.maximum(
        state_norm * target_norm, 1e-30
    )


def direct_fock_fidelity_jax(real_parameters: jnp.ndarray) -> jnp.ndarray:
    return normalized_target_fidelity_from_fock_jax(
        direct_fock_state_jax(real_parameters, FOCK_OPTIMIZER_CUTOFF),
        jnp.asarray(TARGET_FOCK_OPTIMIZER_NUMPY),
    )


def truncated_annihilation_jax(cutoff: int) -> jnp.ndarray:
    """Materialize the cutoff-dimensional annihilation matrix."""
    annihilation = jnp.zeros((cutoff, cutoff), dtype=jnp.complex64)
    indices = jnp.arange(1, cutoff)
    return annihilation.at[indices - 1, indices].set(
        jnp.sqrt(indices).astype(jnp.complex64)
    )


def numerical_displacement_matrix_jax(
    displacement: jnp.ndarray,
    cutoff: int,
) -> jnp.ndarray:
    r"""Return exp(displacement*a_N^dagger-conj(displacement)*a_N).

    This differs from :func:`displacement_matrix_jax`. The analytic recurrence
    returns the upper-left block of the infinite-dimensional displacement.
    Here the ladder operator is truncated first and then exponentiated, so the
    result is exactly unitary inside the finite grid but has a cutoff-dependent
    boundary approximation.
    """
    displacement = displacement.astype(jnp.complex64)
    annihilation = truncated_annihilation_jax(cutoff)
    generator = (
        displacement * annihilation.conj().T
        - jnp.conj(displacement) * annihilation
    )
    return jax.scipy.linalg.expm(generator)


def dense_numerical_fock_state_jax(
    real_parameters: jnp.ndarray,
    cutoff: int,
) -> jnp.ndarray:
    """Propagate a joint 2*cutoff state with materialized dense unitaries."""
    state = jnp.zeros(2 * cutoff, dtype=jnp.complex64)
    state = state.at[0].set(1.0)
    identity = jnp.eye(cutoff, dtype=jnp.complex64)
    parameters = parameters_from_real(real_parameters)
    for layer_index in range(real_parameters.shape[0]):
        layer = parameters[layer_index]
        beta = layer[0]
        rotation = ecd_rotation_jax(
            jnp.real(layer[1]), jnp.real(layer[2])
        ).astype(jnp.complex64)
        rotation_joint = jnp.kron(rotation, identity)
        negative_displacement = numerical_displacement_matrix_jax(
            -beta / 2.0,
            cutoff,
        )
        positive_displacement = numerical_displacement_matrix_jax(
            beta / 2.0,
            cutoff,
        )
        zeros = jnp.zeros_like(identity)
        conditional_displacement = jnp.block(
            [
                [negative_displacement, zeros],
                [zeros, positive_displacement],
            ]
        )
        layer_unitary = conditional_displacement @ rotation_joint
        state = layer_unitary @ state
    return state.reshape(2, cutoff)


def dense_numerical_fock_fidelity_jax(
    real_parameters: jnp.ndarray,
) -> jnp.ndarray:
    return normalized_target_fidelity_from_fock_jax(
        dense_numerical_fock_state_jax(real_parameters, FOCK_OPTIMIZER_CUTOFF),
        jnp.asarray(TARGET_FOCK_OPTIMIZER_NUMPY),
    )


# %% Continuous-anchor and squeezed-packet objectives

def projection_fidelity_jax(
    basis_columns: jnp.ndarray, target_branch: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    gram_matrix = basis_columns.conj().T @ basis_columns
    right_hand_side = basis_columns.conj().T @ target_branch
    ridge = 1e-9 * jnp.eye(gram_matrix.shape[0], dtype=gram_matrix.dtype)
    fitted_coefficients = jnp.linalg.solve(gram_matrix + ridge, right_hand_side)
    approximation = basis_columns @ fitted_coefficients
    overlap = jnp.vdot(target_branch, approximation)
    fidelity = jnp.abs(overlap) ** 2 / jnp.maximum(
        jnp.real(jnp.vdot(target_branch, target_branch))
        * jnp.real(jnp.vdot(approximation, approximation)),
        1e-30,
    )
    return jnp.real(fidelity), fitted_coefficients, approximation


def residual_pivot_indices_jax(
    target_branch: jnp.ndarray,
    candidate_columns: jnp.ndarray,
    retained_count: int,
) -> jnp.ndarray:
    candidate_count = candidate_columns.shape[1]
    orthonormal = jnp.zeros(
        (candidate_columns.shape[0], retained_count),
        dtype=jnp.complex128,
    )
    selected_mask = jnp.zeros(candidate_count, dtype=bool)
    selected_indices = jnp.zeros(retained_count, dtype=jnp.int32)
    residual = target_branch

    def choose_one(selection_index, loop_state):
        current_orthonormal, current_mask, current_indices, current_residual = loop_state
        residualized = candidate_columns - current_orthonormal @ (
            current_orthonormal.conj().T @ candidate_columns
        )
        column_norms = jnp.sum(jnp.abs(residualized) ** 2, axis=0)
        overlaps = residualized.conj().T @ current_residual
        scores = jnp.where(
            (~current_mask) & (column_norms > 1e-24),
            jnp.abs(overlaps) ** 2 / jnp.maximum(column_norms, 1e-30),
            -jnp.inf,
        )
        chosen_index = jnp.asarray(jnp.argmax(scores), dtype=jnp.int32)
        new_vector = residualized[:, chosen_index] / jnp.sqrt(
            jnp.maximum(column_norms[chosen_index], 1e-30)
        )
        current_orthonormal = current_orthonormal.at[:, selection_index].set(
            new_vector
        )
        current_mask = current_mask.at[chosen_index].set(True)
        current_indices = current_indices.at[selection_index].set(chosen_index)
        current_residual = current_residual - new_vector * jnp.vdot(
            new_vector, current_residual
        )
        return current_orthonormal, current_mask, current_indices, current_residual

    _, _, selected_indices, _ = jax.lax.fori_loop(
        0,
        retained_count,
        choose_one,
        (orthonormal, selected_mask, selected_indices, residual),
    )
    return selected_indices


def real_to_complex_jax(real_values: jnp.ndarray) -> jnp.ndarray:
    return real_values[..., 0] + 1j * real_values[..., 1]


def complex_to_real_jax(complex_values: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack((jnp.real(complex_values), jnp.imag(complex_values)), axis=-1)


def coherent_joint_projection_fidelity(
    real_centers: jnp.ndarray, target_state: jnp.ndarray
) -> jnp.ndarray:
    centers = real_to_complex_jax(real_centers)
    approximations = []
    for qubit_branch in range(2):
        basis = coherent_columns_jax(centers[qubit_branch], PACKET_FOCK_CUTOFF)
        _, _, approximation = projection_fidelity_jax(
            basis, target_state[qubit_branch]
        )
        approximations.append(approximation)
    approximation = jnp.stack(approximations)
    overlap = jnp.vdot(target_state, approximation)
    return jnp.real(jnp.abs(overlap) ** 2) / jnp.maximum(
        jnp.real(jnp.vdot(target_state, target_state))
        * jnp.real(jnp.vdot(approximation, approximation)),
        1e-30,
    )


def optimize_centers_jax(
    initial_centers: jnp.ndarray, target_state: jnp.ndarray
) -> jnp.ndarray:
    real_centers = complex_to_real_jax(initial_centers)
    first_moment = jnp.zeros_like(real_centers)
    second_moment = jnp.zeros_like(real_centers)

    def update_center(step_index, loop_state):
        current_centers, moment_one, moment_two = loop_state
        gradient = jax.grad(
            lambda values: -coherent_joint_projection_fidelity(values, target_state)
        )(current_centers)
        moment_one = 0.9 * moment_one + 0.1 * gradient
        moment_two = 0.999 * moment_two + 0.001 * gradient**2
        corrected_one = moment_one / (1.0 - 0.9 ** (step_index + 1))
        corrected_two = moment_two / (1.0 - 0.999 ** (step_index + 1))
        current_centers = current_centers - 0.012 * corrected_one / (
            jnp.sqrt(corrected_two) + 1e-8
        )
        return current_centers, moment_one, moment_two

    real_centers, _, _ = jax.lax.fori_loop(
        0,
        CONTINUOUS_CENTER_STEPS,
        update_center,
        (real_centers, first_moment, second_moment),
    )
    return real_to_complex_jax(real_centers)


def squeezed_columns_jax(
    centers: jnp.ndarray, raw_squeezing: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    maximum_nu = math.sinh(0.8)
    nus = maximum_nu * raw_squeezing / jnp.sqrt(
        1.0 + jnp.abs(raw_squeezing) ** 2
    )
    mus = jnp.sqrt(1.0 + jnp.abs(nus) ** 2)
    gammas = mus * centers + nus * jnp.conj(centers)
    previous = jnp.zeros_like(centers)
    current = jnp.ones_like(centers)
    rows = [current]
    for order in range(PACKET_FOCK_CUTOFF - 1):
        following = (
            gammas * current - nus * math.sqrt(order) * previous
        ) / (mus * math.sqrt(order + 1))
        rows.append(following)
        previous, current = current, following
    columns = jnp.stack(rows, axis=0)
    norms = jnp.sqrt(jnp.sum(jnp.abs(columns) ** 2, axis=0, keepdims=True))
    return columns / jnp.maximum(norms, 1e-30), nus


def squeezed_joint_projection_fidelity(
    real_raw_squeezing: jnp.ndarray,
    centers: jnp.ndarray,
    target_state: jnp.ndarray,
) -> jnp.ndarray:
    raw_squeezing = real_to_complex_jax(real_raw_squeezing)
    approximations = []
    for qubit_branch in range(2):
        basis, _ = squeezed_columns_jax(
            centers[qubit_branch], raw_squeezing[qubit_branch]
        )
        _, _, approximation = projection_fidelity_jax(
            basis, target_state[qubit_branch]
        )
        approximations.append(approximation)
    approximation = jnp.stack(approximations)
    overlap = jnp.vdot(target_state, approximation)
    return jnp.real(jnp.abs(overlap) ** 2) / jnp.maximum(
        jnp.real(jnp.vdot(target_state, target_state))
        * jnp.real(jnp.vdot(approximation, approximation)),
        1e-30,
    )


def optimize_squeezing_jax(
    centers: jnp.ndarray, target_state: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    raw_parameters = jnp.zeros(
        (2, PACKETS_PER_QUBIT_BRANCH, 2), dtype=jnp.float64
    )
    first_moment = jnp.zeros_like(raw_parameters)
    second_moment = jnp.zeros_like(raw_parameters)

    def update_squeezing(step_index, loop_state):
        current_raw, moment_one, moment_two = loop_state
        gradient = jax.grad(
            lambda values: -squeezed_joint_projection_fidelity(
                values, centers, target_state
            )
        )(current_raw)
        moment_one = 0.9 * moment_one + 0.1 * gradient
        moment_two = 0.999 * moment_two + 0.001 * gradient**2
        corrected_one = moment_one / (1.0 - 0.9 ** (step_index + 1))
        corrected_two = moment_two / (1.0 - 0.999 ** (step_index + 1))
        current_raw = current_raw - 0.008 * corrected_one / (
            jnp.sqrt(corrected_two) + 1e-8
        )
        return current_raw, moment_one, moment_two

    raw_parameters, _, _ = jax.lax.fori_loop(
        0,
        SQUEEZE_STEPS,
        update_squeezing,
        (raw_parameters, first_moment, second_moment),
    )
    raw_complex = real_to_complex_jax(raw_parameters)
    _, nus = squeezed_columns_jax(centers[0], raw_complex[0])
    _, nus_one = squeezed_columns_jax(centers[1], raw_complex[1])
    return raw_complex, jnp.stack((nus, nus_one))


def squeezed_canonical_phase_jax(
    centers: jnp.ndarray, nus: jnp.ndarray
) -> jnp.ndarray:
    mus = jnp.sqrt(1.0 + jnp.abs(nus) ** 2)
    logarithm = -0.5 * jnp.abs(centers) ** 2 - (
        nus * jnp.conj(centers) ** 2 / (2.0 * mus)
    )
    return jnp.imag(logarithm)


def compress_packet_state_jax(
    target_state: jnp.ndarray,
    candidate_coefficients: tuple[jnp.ndarray, jnp.ndarray],
    candidate_centers: tuple[jnp.ndarray, jnp.ndarray],
    method_name: str,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    stopped_target = jax.lax.stop_gradient(target_state)
    seed_centers = []
    for qubit_branch in range(2):
        candidate_basis = coherent_columns_jax(
            candidate_centers[qubit_branch], PACKET_FOCK_CUTOFF
        )
        selected_indices = residual_pivot_indices_jax(
            stopped_target[qubit_branch],
            candidate_basis,
            PACKETS_PER_QUBIT_BRANCH,
        )
        seed_centers.append(candidate_centers[qubit_branch][selected_indices])
    optimized_centers = optimize_centers_jax(
        jax.lax.stop_gradient(jnp.stack(seed_centers)), stopped_target
    )
    optimized_centers = jax.lax.stop_gradient(optimized_centers)

    fitted_coefficients = []
    projected_branches = []
    output_nus = []
    if method_name == "continuous_anchors":
        for qubit_branch in range(2):
            basis = coherent_columns_jax(
                optimized_centers[qubit_branch], PACKET_FOCK_CUTOFF
            )
            _, coefficients, approximation = projection_fidelity_jax(
                basis, stopped_target[qubit_branch]
            )
            fitted_coefficients.append(coefficients)
            projected_branches.append(approximation)
            output_nus.append(jnp.zeros_like(optimized_centers[qubit_branch]))
    elif method_name == "squeezed_packets":
        raw_squeezing, nus = optimize_squeezing_jax(
            optimized_centers, stopped_target
        )
        raw_squeezing = jax.lax.stop_gradient(raw_squeezing)
        nus = jax.lax.stop_gradient(nus)
        for qubit_branch in range(2):
            basis, _ = squeezed_columns_jax(
                optimized_centers[qubit_branch], raw_squeezing[qubit_branch]
            )
            _, coefficients, approximation = projection_fidelity_jax(
                basis, stopped_target[qubit_branch]
            )
            fitted_coefficients.append(coefficients)
            projected_branches.append(approximation)
            output_nus.append(nus[qubit_branch])
    else:
        raise ValueError(method_name)

    projected_state = jax.lax.stop_gradient(jnp.stack(projected_branches))
    # Straight-through derivative: forward propagation uses the compressed
    # state, while the outer gradient treats the projection as the identity.
    differentiable_state = target_state + jax.lax.stop_gradient(
        projected_state - target_state
    )
    return (
        differentiable_state,
        (jax.lax.stop_gradient(fitted_coefficients[0]), jax.lax.stop_gradient(fitted_coefficients[1])),
        (optimized_centers[0], optimized_centers[1]),
        (jax.lax.stop_gradient(output_nus[0]), jax.lax.stop_gradient(output_nus[1])),
    )


def packet_fidelity_jax(
    real_parameters: jnp.ndarray, method_name: str
) -> jnp.ndarray:
    parameters = parameters_from_real(real_parameters)
    state = jnp.zeros((2, PACKET_FOCK_CUTOFF), dtype=jnp.complex128)
    state = state.at[0, 0].set(1.0)
    state = fock_step_jax(state, parameters[0], PACKET_FOCK_CUTOFF)

    first_beta = parameters[0, 0]
    first_rotation = ecd_rotation_jax(
        jnp.real(parameters[0, 1]), jnp.real(parameters[0, 2])
    )
    coefficients = (
        jnp.asarray([first_rotation[0, 0]]),
        jnp.asarray([first_rotation[1, 0]]),
    )
    centers = (
        jnp.asarray([-first_beta / 2.0]),
        jnp.asarray([first_beta / 2.0]),
    )
    nus = (jnp.zeros(1, dtype=jnp.complex128), jnp.zeros(1, dtype=jnp.complex128))

    for layer_index in range(1, real_parameters.shape[0]):
        layer = parameters[layer_index]
        beta = layer[0]
        rotation = ecd_rotation_jax(jnp.real(layer[1]), jnp.real(layer[2]))
        output_coefficients = []
        output_centers = []
        output_nus = []
        for output_qubit, shift in enumerate((-beta / 2.0, beta / 2.0)):
            coefficient_parts = []
            center_parts = []
            nu_parts = []
            for input_qubit in range(2):
                phase = jnp.exp(-1j * symplectic_jax(shift, centers[input_qubit]))
                if method_name == "squeezed_packets":
                    phase = phase * jnp.exp(
                        1j
                        * (
                            squeezed_canonical_phase_jax(
                                centers[input_qubit] + shift, nus[input_qubit]
                            )
                            - squeezed_canonical_phase_jax(
                                centers[input_qubit], nus[input_qubit]
                            )
                        )
                    )
                coefficient_parts.append(
                    rotation[output_qubit, input_qubit]
                    * coefficients[input_qubit]
                    * phase
                )
                center_parts.append(centers[input_qubit] + shift)
                nu_parts.append(nus[input_qubit])
            output_coefficients.append(jnp.concatenate(coefficient_parts))
            output_centers.append(jnp.concatenate(center_parts))
            output_nus.append(jnp.concatenate(nu_parts))
        coefficients = (output_coefficients[0], output_coefficients[1])
        centers = (output_centers[0], output_centers[1])
        nus = (output_nus[0], output_nus[1])

        state = fock_step_jax(state, layer, PACKET_FOCK_CUTOFF)
        if coefficients[0].shape[0] > PACKETS_PER_QUBIT_BRANCH:
            state, coefficients, centers, nus = compress_packet_state_jax(
                state, coefficients, centers, method_name
            )
    return normalized_target_fidelity_from_fock_jax(
        state,
        jnp.asarray(TARGET_FOCK_PACKET_NUMPY),
    )


# %% Matched multirestart optimization

def deterministic_initial_parameters(layer_count: int, restart_index: int) -> np.ndarray:
    """Return one reproducible draw from the user-specified initialization.

    ``beta`` is circular complex normal with E[beta] = 0 and
    E[|beta|**2] = 0.5. Therefore its real and imaginary components are
    independent N(0, 0.25). Both optimized angles ``phi`` and ``theta`` are
    independent Uniform[0, 2*pi]. The experiment fixes ``gamma = 0`` exactly,
    leaving four optimized real numbers per layer.
    """
    random_generator = np.random.default_rng(
        INITIALIZATION_SEED_BASE + 97 * layer_count + restart_index
    )
    return np.column_stack(
        (
            random_generator.normal(
                scale=INITIAL_COMPONENT_STANDARD_DEVIATION,
                size=layer_count,
            ),
            random_generator.normal(
                scale=INITIAL_COMPONENT_STANDARD_DEVIATION,
                size=layer_count,
            ),
            random_generator.uniform(0.0, 2.0 * math.pi, size=layer_count),
            random_generator.uniform(0.0, 2.0 * math.pi, size=layer_count),
        )
    )


def fidelity_function_for_method(method_name: str):
    if method_name == "full_coherent":
        return full_coherent_fidelity_jax
    if method_name == "direct_fock":
        return direct_fock_fidelity_jax
    if method_name == "dense_numerical_fock":
        return dense_numerical_fock_fidelity_jax
    if method_name in ("continuous_anchors", "squeezed_packets"):
        return lambda values: packet_fidelity_jax(values, method_name)
    raise ValueError(method_name)


def run_optimizer_restart(
    initial_parameters: np.ndarray,
    fidelity_function,
    outer_steps: int = OUTER_STEPS,
) -> tuple[np.ndarray, float]:
    loss_function = lambda values: 1.0 - fidelity_function(values)
    learning_rate_schedule = optax.piecewise_constant_schedule(
        OUTER_LEARNING_RATE,
        {int(0.75 * outer_steps): 0.25},
    )
    optimizer = optax.adam(learning_rate_schedule)
    initial = jnp.asarray(initial_parameters, dtype=jnp.float64)
    initial_optimizer_state = optimizer.init(initial)

    @jax.jit
    def train_all_steps(parameters, optimizer_state):
        initial_loss = loss_function(parameters)

        def train_one_step(_, loop_state):
            current_parameters, current_optimizer_state, best_parameters, best_loss = loop_state
            current_loss, gradients = jax.value_and_grad(loss_function)(current_parameters)
            evaluated_parameters = current_parameters
            updates, current_optimizer_state = optimizer.update(
                gradients, current_optimizer_state, current_parameters
            )
            current_parameters = optax.apply_updates(current_parameters, updates)
            current_parameters = current_parameters.at[:, :2].set(
                jnp.clip(
                    current_parameters[:, :2],
                    -PARAMETER_ABSOLUTE_LIMIT,
                    PARAMETER_ABSOLUTE_LIMIT,
                )
            )
            improved = current_loss < best_loss
            best_parameters = jnp.where(improved, evaluated_parameters, best_parameters)
            best_loss = jnp.minimum(best_loss, current_loss)
            return current_parameters, current_optimizer_state, best_parameters, best_loss

        _, _, best_parameters, best_loss = jax.lax.fori_loop(
            0,
            outer_steps,
            train_one_step,
            (parameters, optimizer_state, parameters, initial_loss),
        )
        final_loss = loss_function(best_parameters)
        return best_parameters, jnp.minimum(best_loss, final_loss)

    best_parameters, best_loss = train_all_steps(initial, initial_optimizer_state)
    jax.block_until_ready(best_parameters)
    return np.asarray(best_parameters), float(best_loss)


def coherax_gate_time_microseconds(real_parameters: np.ndarray) -> float:
    beta_magnitudes = np.linalg.norm(np.asarray(real_parameters)[:, :2], axis=1)
    displacement_seconds = np.maximum(
        beta_magnitudes / (COHERAX_CHI_RADIANS_PER_SECOND * COHERAX_DRIVE_SCALE),
        COHERAX_MINIMUM_DISPLACEMENT_SECONDS,
    )
    return float(
        np.sum(COHERAX_ANCILLA_SECONDS + displacement_seconds) * 1e6
    )


def eickbusch_gate_time_lower_bound_microseconds(real_parameters: np.ndarray) -> float:
    """Layerwise lower bound, not the unavailable upstream pulse compiler."""
    beta_magnitudes = np.linalg.norm(np.asarray(real_parameters)[:, :2], axis=1)
    interaction_seconds = beta_magnitudes / (
        EICKBUSCH_CHI_RADIANS_PER_SECOND * EICKBUSCH_DRIVE_SCALE
    )
    return float(
        np.sum(np.maximum(interaction_seconds, EICKBUSCH_LAYER_FLOOR_SECONDS))
        * 1e6
    )


def sequence_json(real_parameters: np.ndarray) -> list[dict[str, float]]:
    return [
        {
            "layer": layer_index + 1,
            "beta_real": float(layer[0]),
            "beta_imag": float(layer[1]),
            "phi_radians": float(layer[2]),
            "theta_radians": float(layer[3]),
            "gamma_radians": 0.0,
        }
        for layer_index, layer in enumerate(np.asarray(real_parameters))
    ]


def run_all_optimizations(
    layer_counts: tuple[int, ...] = LAYER_COUNTS,
    method_names: tuple[str, ...] = METHOD_NAMES,
) -> tuple[list[dict], dict[tuple[str, int], np.ndarray]]:
    records = []
    winning_parameters = {}
    for method_name in method_names:
        for layer_count in layer_counts:
            method_depth_started = time.perf_counter()
            restart_records = []
            fidelity_function = fidelity_function_for_method(method_name)
            outer_steps = OUTER_STEPS
            for restart_index in range(RESTART_COUNT):
                initial = deterministic_initial_parameters(layer_count, restart_index)
                restart_started = time.perf_counter()
                optimized, native_loss = run_optimizer_restart(
                    initial, fidelity_function, outer_steps=outer_steps
                )
                restart_seconds = time.perf_counter() - restart_started
                exact_fidelity = exact_coherent_score_numpy(optimized)
                restart_record = {
                    "restart": restart_index,
                    "wall_seconds": restart_seconds,
                    "native_fidelity": float(np.clip(1.0 - native_loss, 0.0, 1.0)),
                    "exact_fidelity": exact_fidelity,
                    "exact_infidelity": 1.0 - exact_fidelity,
                    "parameters": optimized,
                }
                restart_records.append(restart_record)
                print(
                    "RESTART",
                    method_name,
                    layer_count,
                    restart_index,
                    f"native_F={restart_record['native_fidelity']:.10f}",
                    f"exact_F={exact_fidelity:.10f}",
                    f"seconds={restart_seconds:.3f}",
                    flush=True,
                )
            total_seconds = time.perf_counter() - method_depth_started
            winning_restart = max(
                restart_records, key=lambda item: item["exact_fidelity"]
            )
            parameters = winning_restart.pop("parameters")
            for other_restart in restart_records:
                other_restart.pop("parameters", None)
            winning_parameters[(method_name, layer_count)] = parameters
            record = {
                "method": method_name,
                "method_label": METHOD_LABELS[method_name],
                "layers": layer_count,
                "restart_count": RESTART_COUNT,
                "outer_steps_per_restart": outer_steps,
                "total_multirestart_wall_seconds": total_seconds,
                "winning_restart": winning_restart["restart"],
                "winning_restart_wall_seconds": winning_restart["wall_seconds"],
                "native_fidelity": winning_restart["native_fidelity"],
                "exact_fidelity": winning_restart["exact_fidelity"],
                "exact_infidelity": winning_restart["exact_infidelity"],
                "coherax_gate_time_microseconds": coherax_gate_time_microseconds(parameters),
                "eickbusch_lower_bound_microseconds": eickbusch_gate_time_lower_bound_microseconds(parameters),
                "restarts": restart_records,
            }
            records.append(record)
            with (SEQUENCE_DIRECTORY / f"{method_name}_n{layer_count}.json").open(
                "w", encoding="utf-8"
            ) as sequence_file:
                json.dump(
                    {
                        "method": method_name,
                        "layers": layer_count,
                        "target": (
                            "logical |0> finite hex GKP, Delta=0.2, "
                            f"complete hex shell radius {TARGET_HEX_SHELL_RADIUS}"
                        ),
                        "exact_fidelity": record["exact_fidelity"],
                        "sequence": sequence_json(parameters),
                    },
                    sequence_file,
                    indent=2,
                )
            print("WINNER", json.dumps({k: v for k, v in record.items() if k != "restarts"}), flush=True)
    return records, winning_parameters


# %% Exact validation, plots, and saved artifacts

def exact_output_fock_numpy(real_parameters: np.ndarray, cutoff: int) -> np.ndarray:
    """Convert the exact coherent-path output to a finite Fock vector for plots."""
    real_parameters = np.asarray(real_parameters, dtype=np.float64)
    coefficients = [np.asarray([1.0 + 0.0j]), np.empty(0, dtype=np.complex128)]
    centers = [np.asarray([0.0 + 0.0j]), np.empty(0, dtype=np.complex128)]
    for beta_real, beta_imag, phi, theta in real_parameters:
        beta = complex(beta_real, beta_imag)
        prefactor = np.exp(-0.5j * phi)
        rotation = prefactor * np.asarray(
            [
                [
                    np.sin(theta / 2.0) * np.exp(1j * phi),
                    np.cos(theta / 2.0) * np.exp(1j * phi),
                ],
                [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
            ],
            dtype=np.complex128,
        )
        output_coefficients = []
        output_centers = []
        for output_qubit, shift in enumerate((-beta / 2.0, beta / 2.0)):
            coefficient_parts = []
            center_parts = []
            for input_qubit in range(2):
                if coefficients[input_qubit].size == 0:
                    continue
                symplectic = (
                    shift.real * centers[input_qubit].imag
                    - shift.imag * centers[input_qubit].real
                )
                coefficient_parts.append(
                    rotation[output_qubit, input_qubit]
                    * coefficients[input_qubit]
                    * np.exp(-1j * symplectic)
                )
                center_parts.append(centers[input_qubit] + shift)
            output_coefficients.append(np.concatenate(coefficient_parts))
            output_centers.append(np.concatenate(center_parts))
        coefficients, centers = output_coefficients, output_centers
    branches = []
    for qubit_branch in range(2):
        branch_centers = centers[qubit_branch]
        current_columns = np.exp(-0.5 * np.abs(branch_centers) ** 2).astype(
            np.complex128
        )
        branch = np.empty(cutoff, dtype=np.complex128)
        branch[0] = np.sum(current_columns * coefficients[qubit_branch])
        for fock_number in range(1, cutoff):
            current_columns = (
                current_columns * branch_centers / math.sqrt(fock_number)
            )
            branch[fock_number] = np.sum(
                current_columns * coefficients[qubit_branch]
            )
        branches.append(branch)
    return np.stack(branches)


def reduced_oscillator_density_numpy(
    real_parameters: np.ndarray, cutoff: int
) -> np.ndarray:
    state = exact_output_fock_numpy(real_parameters, cutoff)
    density = sum(np.outer(branch, np.conj(branch)) for branch in state)
    trace = float(np.trace(density).real)
    return density / trace


def add_line_plot(
    records: list[dict],
    value_name: str,
    ylabel: str,
    filename: str,
    logarithmic: bool = False,
) -> None:
    figure, axis = plt.subplots(figsize=(6.6, 4.2))
    for method_name in METHOD_NAMES:
        selected = sorted(
            [record for record in records if record["method"] == method_name],
            key=lambda item: item["layers"],
        )
        values = np.asarray([record[value_name] for record in selected])
        if logarithmic:
            values = np.maximum(values, 1e-15)
        axis.plot(
            [record["layers"] for record in selected],
            values,
            marker="o",
            linewidth=2,
            color=METHOD_COLORS[method_name],
            label=METHOD_LABELS[method_name],
        )
    axis.set_xlabel("number of ECD-plus-rotation layers, n")
    axis.xaxis.labelpad = 9
    axis.set_ylabel(ylabel)
    axis.set_xticks(LAYER_COUNTS)
    if logarithmic:
        axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout(pad=1.4)
    figure.savefig(FIGURE_DIRECTORY / filename, dpi=190, bbox_inches="tight")
    plt.close(figure)


def make_wigner_plate(
    winning_parameters: dict[tuple[str, int], np.ndarray]
) -> None:
    depth_count = len(LAYER_COUNTS)
    method_count = len(METHOD_NAMES)
    figure = plt.figure(figsize=(3.05 * depth_count, 3.0 * (method_count + 1)))
    grid = figure.add_gridspec(
        method_count + 1,
        depth_count,
        height_ratios=tuple([1.0] * method_count + [1.12]),
        hspace=0.28,
        wspace=0.12,
    )
    for method_row, method_name in enumerate(METHOD_NAMES):
        for depth_column, layer_count in enumerate(LAYER_COUNTS):
            axis = figure.add_subplot(grid[method_row, depth_column])
            density = reduced_oscillator_density_numpy(
                winning_parameters[(method_name, layer_count)], WIGNER_FOCK_CUTOFF
            )
            dq.plot.wigner(
                jnp.asarray(density),
                ax=axis,
                xmax=5.2,
                vmax=2.0 / math.pi,
                npixels=121,
                colorbar=False,
                cross=True,
            )
            if method_row == 0:
                axis.set_title(f"n = {layer_count}", fontsize=10)
            if depth_column == 0:
                axis.set_ylabel(METHOD_LABELS[method_name], fontsize=9)
            else:
                axis.set_ylabel("")
            axis.set_xlabel("")
    reference_axis = figure.add_subplot(grid[method_count, :])
    target_ket = TARGET_FOCK_WIGNER_NUMPY.copy()
    target_ket = target_ket / np.linalg.norm(target_ket)
    dq.plot.wigner(
        jnp.asarray(target_ket[:, None]),
        ax=reference_axis,
        xmax=5.2,
        vmax=2.0 / math.pi,
        npixels=161,
        colorbar=True,
        cross=True,
    )
    reference_axis.set_title(
        r"exact finite hex-GKP target: logical $|0_L\rangle$, $\Delta=0.2$",
        fontsize=11,
    )
    figure.suptitle(
        "Wigner functions of the winning physical circuits",
        fontsize=14,
        y=0.995,
    )
    figure.savefig(
        FIGURE_DIRECTORY / "wigner_optimizer_grid.png",
        dpi=175,
        bbox_inches="tight",
    )
    plt.close(figure)


def write_experiment_artifacts(
    records: list[dict],
    winning_parameters: dict[tuple[str, int], np.ndarray],
) -> None:
    records = sorted(
        records,
        key=lambda record: (
            METHOD_NAMES.index(record["method"]),
            LAYER_COUNTS.index(record["layers"]),
        ),
    )
    validation_records = []
    for record in records:
        parameters = winning_parameters[(record["method"], record["layers"])]
        validation_state = exact_output_fock_numpy(parameters, VALIDATION_FOCK_CUTOFF)
        validation_norm = float(np.vdot(validation_state, validation_state).real)
        target_amplitudes = np.sum(
            np.conj(TARGET_FOCK_VALIDATION_NUMPY)[None, :] * validation_state,
            axis=1,
        )
        target_validation_norm = float(
            np.vdot(
                TARGET_FOCK_VALIDATION_NUMPY,
                TARGET_FOCK_VALIDATION_NUMPY,
            ).real
        )
        validation_fidelity = float(
            np.sum(np.abs(target_amplitudes) ** 2)
            / (validation_norm * target_validation_norm)
        )
        validation = {
            "method": record["method"],
            "layers": record["layers"],
            "fock_cutoff": VALIDATION_FOCK_CUTOFF,
            "projected_norm": validation_norm,
            "target_projected_norm": target_validation_norm,
            "fock_fidelity": validation_fidelity,
            "absolute_difference_from_exact_analytic": abs(
                validation_fidelity - record["exact_fidelity"]
            ),
        }
        validation_records.append(validation)

    add_line_plot(
        records,
        "total_multirestart_wall_seconds",
        "total wall time for all restarts (s)",
        "wall_time_vs_layers.png",
    )
    add_line_plot(
        records,
        "exact_infidelity",
        r"exact infidelity, $1-F$",
        "infidelity_vs_layers.png",
        logarithmic=True,
    )
    add_line_plot(
        records,
        "coherax_gate_time_microseconds",
        r"Coherax gate-time estimate, $T_C$ ($\mu$s)",
        "gate_time_coherax_vs_layers.png",
    )
    add_line_plot(
        records,
        "eickbusch_lower_bound_microseconds",
        r"Eickbusch-hardware lower bound, $T_{E,\mathrm{LB}}$ ($\mu$s)",
        "gate_time_eickbusch_bound_vs_layers.png",
    )
    make_wigner_plate(winning_parameters)

    archive = {
        "layer_counts": np.asarray(LAYER_COUNTS),
        "method_names": np.asarray(METHOD_NAMES),
    }
    for (method_name, layer_count), parameters in winning_parameters.items():
        archive[f"{method_name}_n{layer_count}_real_parameters"] = parameters
        archive[f"{method_name}_n{layer_count}_complex_parameters"] = np.asarray(
            parameters_from_real(jnp.asarray(parameters))
        )
    np.savez(EXPERIMENT_DIRECTORY / "best_sequences.npz", **archive)

    results = {
        "experiment": "07_hex_gkp_state_preparation_optimizers",
        "target": {
            "formula": (
                "normalized sum over complete hex-shell sites d=m*alpha+l*beta "
                "with m even and coefficient exp[-i*pi*k*l-Delta^2*|d|^2]"
            ),
            "logical_index": TARGET_LOGICAL_INDEX,
            "Delta": TARGET_DELTA,
            "lattice_alpha": "sqrt(pi/sqrt(3))",
            "lattice_beta": "exp(2*pi*i/3)*alpha",
            "hex_shell_radius": TARGET_HEX_SHELL_RADIUS,
            "coherent_site_count": int(TARGET_CENTERS_NUMPY.size),
            "mean_photon_number_from_validation_projection": TARGET_MEAN_PHOTON_VALIDATION,
            "shell_reference_radius": TARGET_CONVERGENCE_REFERENCE_RADIUS,
            "fidelity_with_reference_shell": float(TARGET_SHELL_REFERENCE_FIDELITY),
            "fock_projected_norms": {
                str(cutoff): float(
                    np.vdot(
                        coherent_superposition_fock_numpy(
                            TARGET_COEFFICIENTS_NUMPY,
                            TARGET_CENTERS_NUMPY,
                            cutoff,
                        ),
                        coherent_superposition_fock_numpy(
                            TARGET_COEFFICIENTS_NUMPY,
                            TARGET_CENTERS_NUMPY,
                            cutoff,
                        ),
                    ).real
                )
                for cutoff in (
                    FOCK_OPTIMIZER_CUTOFF,
                    WIGNER_FOCK_CUTOFF,
                    VALIDATION_FOCK_CUTOFF,
                )
            },
            "initial_joint_state": "|qubit 0> tensor |oscillator 0>",
            "fidelity": "sum_q |<target|psi_q>|^2 for the exact joint pure circuit output",
        },
        "configuration": {
            "layer_counts": list(LAYER_COUNTS),
            "methods": list(METHOD_NAMES),
            "restart_count": RESTART_COUNT,
            "outer_steps_by_depth": {
                str(layer_count): OUTER_STEPS for layer_count in LAYER_COUNTS
            },
            "outer_learning_rate": OUTER_LEARNING_RATE,
            "initialization": {
                "beta_distribution": "iid circular complex Gaussian",
                "beta_mean": 0.0,
                "beta_complex_variance_E_abs_squared": INITIAL_COMPLEX_VARIANCE,
                "beta_real_component_variance": INITIAL_COMPLEX_VARIANCE / 2.0,
                "beta_imag_component_variance": INITIAL_COMPLEX_VARIANCE / 2.0,
                "phi_distribution": "iid Uniform[0, 2*pi]",
                "theta_distribution": "iid Uniform[0, 2*pi]",
                "gamma_radians": 0.0,
                "seed_base": INITIALIZATION_SEED_BASE,
            },
            "fock_optimizer_cutoff": FOCK_OPTIMIZER_CUTOFF,
            "full_coherent_training_target": (
                "normalized target projection onto Fock levels 0 through 47; "
                "the coherent circuit state itself is not truncated"
            ),
            "analytic_fock_displacement": (
                "upper-left block of the exact infinite-dimensional "
                "displacement, built by recurrence"
            ),
            "dense_numerical_fock_displacement": (
                "expm(alpha*a_N^dagger-conjugate(alpha)*a_N) after truncating a_N"
            ),
            "dense_numerical_fock_joint_unitary_shape": [
                2 * FOCK_OPTIMIZER_CUTOFF,
                2 * FOCK_OPTIMIZER_CUTOFF,
            ],
            "dense_numerical_fock_precision": (
                "complex64 matrix exponentials and joint-state propagation; "
                "target-overlap arithmetic may promote; float64 exact rescoring"
            ),
            "packet_fock_cutoff": PACKET_FOCK_CUTOFF,
            "packets_per_qubit_branch": PACKETS_PER_QUBIT_BRANCH,
            "total_retained_packets": 2 * PACKETS_PER_QUBIT_BRANCH,
            "continuous_center_steps": CONTINUOUS_CENTER_STEPS,
            "squeeze_steps": SQUEEZE_STEPS,
            "validation_fock_cutoff": VALIDATION_FOCK_CUTOFF,
            "outer_gradient_for_packet_methods": "straight-through projection estimator",
            "winner_selection": "largest exact analytic fidelity, never native surrogate fidelity",
        },
        "timing_definitions": {
            "optimization_wall_time": "elapsed time for all restarts at one method and depth, including first JAX compilation and exact host rescoring",
            "coherax_gate_time_microseconds": "sum_i [24 ns + max(|beta_i|/((2*pi*50 kHz)*20), 48 ns)]",
            "eickbusch_lower_bound_microseconds": "sum_i max(|beta_i|/((2*pi*33 kHz)*30), 224 ns); this is not an upstream pulse-compiler duration",
        },
        "records": records,
        "validations": validation_records,
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jax_devices": [str(device) for device in jax.devices()],
            "dynamiqs": getattr(dq, "__version__", "unknown"),
            "platform": platform.platform(),
        },
    }
    with (EXPERIMENT_DIRECTORY / "results.json").open("w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2)
    print("EXPERIMENT_COMPLETE", json.dumps(results["configuration"]), flush=True)


def run_complete_experiment() -> None:
    if os.environ.get("COHERAX_EXP07_REUSE_RESULTS") == "1":
        results_path = EXPERIMENT_DIRECTORY / "results.json"
        archive_path = EXPERIMENT_DIRECTORY / "best_sequences.npz"
        if not results_path.exists() or not archive_path.exists():
            raise FileNotFoundError(
                "COHERAX_EXP07_REUSE_RESULTS=1 requires existing results.json "
                "and best_sequences.npz"
            )
        with results_path.open(encoding="utf-8") as results_file:
            saved_results = json.load(results_file)
        expected_pairs = {
            (method_name, layer_count)
            for method_name in METHOD_NAMES
            for layer_count in LAYER_COUNTS
        }
        saved_pairs = {
            (record["method"], record["layers"])
            for record in saved_results["records"]
        }
        if saved_pairs != expected_pairs:
            raise ValueError(
                "Saved results do not contain exactly the configured "
                "method/depth pairs"
            )
        with np.load(archive_path) as archive:
            missing_keys = [
                f"{method_name}_n{layer_count}_real_parameters"
                for method_name, layer_count in sorted(expected_pairs)
                if f"{method_name}_n{layer_count}_real_parameters" not in archive
            ]
        if missing_keys:
            raise ValueError(f"Saved sequence archive is missing {missing_keys}")
        print(
            "EXPERIMENT_REUSED_EXISTING_RESULTS",
            json.dumps(saved_results["configuration"]),
            flush=True,
        )
        return
    records, winning_parameters = run_all_optimizations()
    write_experiment_artifacts(records, winning_parameters)


if __name__ == "__main__":
    run_complete_experiment()
