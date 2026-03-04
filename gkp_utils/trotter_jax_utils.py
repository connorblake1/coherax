from gkp_utils.trotter_graph_utils import *
from gkp_utils.trotter_utils import PrimitiveGenerator, GateSequence
from gkp_utils.utils import (
    p,
    x,
    IN,
    I2,
    sigma_x,
    sigma_y,
    sigma_z,
    dqtensor,
    dqdag,
    GKP_N,
    dqptrace,
    dqtrace,
    dqfock_dm,
)
from jaxtyping import Array
import jax.scipy.linalg as jla


def concretize_sequence(sequence: GateSequence, swap_dict: dict):
    new_sequence_ops = []
    for op in sequence.ops:
        s, i, v = op.get_items()
        if isinstance(v, (int, float, complex)):
            new_val = v
        else:
            new_val = complex(v.evalf(subs=swap_dict))
        new_sequence_ops.append(PrimitiveGenerator(s, i, new_val))
    if isinstance(sequence.cost, (int, float, complex)):
        new_cost = sequence.cost
    else:
        new_cost = complex(sequence.cost.evalf(subs=swap_dict))
    return GateSequence(ops=new_sequence_ops, cost=new_cost)


def prim_generator_to_unitary(gen: PrimitiveGenerator):
    op_dict = {
        "p": p,
        "q": x,
        "": IN,
    }
    qubit_dict = {0: I2, 1: sigma_x, 2: sigma_y, 3: sigma_z}
    return jla.expm(
        -1j * complex(gen.val) * dqtensor(op_dict[gen.seq], qubit_dict[gen.ind])
    )


def sequence_to_unitary(sequence: GateSequence) -> Array:
    out = dqtensor(IN, I2)
    for prim in sequence.ops:
        out = out @ prim_generator_to_unitary(prim)
    return out


def cycle_qubits(rho, gate_q, gate_p):
    rho_raised = dqtensor(rho, dqfock_dm(2, 1))  # TODO phase
    rho_evolved = gate_q @ rho_raised @ dqdag(gate_q)
    rho = dqptrace(rho_evolved, 0, (GKP_N, 2))
    rho = rho / dqtrace(rho)

    rho_raised = dqtensor(rho, dqfock_dm(2, 0))  # TODO phase
    rho_evolved = gate_p @ rho_raised @ dqdag(gate_p)
    rho = dqptrace(rho_evolved, 0, (GKP_N, 2))

    return rho / dqtrace(rho)


def cycle_N_times(rho, N, gate_q, gate_p):
    for _ in range(N):
        rho = cycle_qubits(rho, gate_q=gate_q, gate_p=gate_p)
    return rho
