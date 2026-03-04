from gkp_utils.trotter_utils import GeneratorSum, GateSequence, PrimitiveGenerator
import sympy as sp
import itertools
from typing import List

I = sp.I


def build_accessible_graph_singleonly(primitive_keys, printing=False, cycles=4):
    # where [A,B] = C where C is a single q^i p^j \sigma_\alpha
    commutation_dict = {}
    lookup_dict = {}
    all_accessible = primitive_keys.copy()
    for _ in range(cycles):
        for k1 in all_accessible:
            for k2 in all_accessible:
                com_value = (
                    GeneratorSum(*k1).commutator(GeneratorSum(*k2)).canonicalize()
                )
                to_unpack = len(com_value.qubit_indices)
                if to_unpack == 2:
                    # identity commutes with all
                    if com_value.qp_sequences[0] == "":
                        com_value.qp_sequences.pop(0)
                        com_value.qubit_indices.pop(0)
                        com_value.values.pop(0)
                    elif com_value.qp_sequences[1] == "":
                        com_value.qp_sequences.pop(1)
                        com_value.qubit_indices.pop(1)
                        com_value.values.pop(1)
                    else:
                        continue
                if to_unpack > 2:
                    # print(k1, k2, com_value.qp_sequences, com_value.qubit_indices, com_value.values)
                    continue
                elif to_unpack == 0 or com_value.values[0] == 0:
                    continue
                new_key = (com_value.qp_sequences[0], com_value.qubit_indices[0])
                commutation_dict[(k1, k2)] = new_key
                commutation_dict[(k2, k1)] = new_key
                possible_parent_list = lookup_dict.get(new_key, [])
                if (k1, k2) not in possible_parent_list and (
                    k2,
                    k1,
                ) not in possible_parent_list:
                    possible_parent_list.append((k1, k2))
                lookup_dict[new_key] = possible_parent_list
        added_keys = [v for k, v in commutation_dict.items() if v not in all_accessible]
        # print("cycle", c, added_keys)
        all_accessible.extend(added_keys)
    return commutation_dict, lookup_dict, all_accessible


def compute_min_depth_decompositions(primitive_keys, preimage):
    # compute the minimum depth path needed to construct key k
    # preimage is the lookup dict for [k_1, k_2] = k so preimage(k) contains all the sets like (k_1, k_2)
    depths = {p: 0 for p in primitive_keys}
    decomp = {}

    while True:
        updated = False
        for x, pairs in preimage.items():
            if x in depths:
                continue
            for a, b in pairs:
                if a in depths and b in depths:
                    depths[x] = max(depths[a], depths[b]) + 1
                    decomp[x] = (a, b)
                    updated = True
                    break
        if not updated:
            break

    return decomp, depths


def decompose_key_path(k, primitive_keys, decomps, depths):
    if k not in depths:
        raise KeyError(f"No generation path for {k}")

    needed = set()

    def collect(x):
        if x in needed:
            return
        needed.add(x)
        if x in primitive_keys:
            return
        a, b = decomps[x]
        collect(a)
        collect(b)

    collect(k)
    return depths[k], needed, decomps


def forward_pass(k, decomps, primitive_keys):
    # compute the required forward pass to construct key k
    visited = set()
    order = []

    def build(x):
        if x in primitive_keys or x in visited:
            return
        a, b = decomps[x]
        build(a)
        build(b)
        order.append((a, b, x))
        visited.add(x)

    build(k)
    return order


def batch_forward_pass(ks, decomps, primitive_keys):
    total = []
    for k in ks:
        total = total + forward_pass(k, decomps=decomps, primitive_keys=primitive_keys)
    return total


def key_commutator(k1, k2):
    return GeneratorSum(*k1).commutator(GeneratorSum(*k2)).canonicalize()


def find_index_by_third(lst, k) -> int:
    for i, (_, _, val) in enumerate(lst):
        if val == k:
            return i
    raise ValueError(f"{k!r} not found in third position of any tuple")


def get_commutator_sequence(k, instrs, primitive_keys, value) -> GateSequence:
    if k in primitive_keys:
        return GateSequence([PrimitiveGenerator(*k, value)], 0)
    i = find_index_by_third(instrs, k)
    prim_AC, prim_BD = instrs[i][0], instrs[i][1]
    implied_sum = (
        GeneratorSum(*prim_AC, 1).commutator(GeneratorSum(*prim_BD, 1)).canonicalize()
    )
    # print(prim_AC, prim_BD, *implied_sum.lists())
    implied_seq, implied_qubits, implied_value = implied_sum.lists()
    extra_ops = None
    alpha = value
    if len(implied_seq) > 1:
        if len(implied_seq) == 2:
            # print(f"{implied_seq=}")
            float_ind = None
            if implied_seq[0] == "":
                float_ind = 0
            elif implied_seq[1] == "":
                float_ind = 1
            if float_ind is None:
                raise NotImplementedError("Too many terms for double")
            beta = implied_value[1 - float_ind]
            gamma = implied_value[float_ind]
            # print(f"{alpha=}, {beta=}, {gamma=}")
            if implied_qubits[0] == implied_qubits[1]:
                # print("extra op", -alpha * gamma / beta / (-sp.I))
                # this adds an implicit noncommutative summation splitting while assuming it's the lower weight operarator
                # See Jiang2 p3 for derivation
                extra_ops = GateSequence(
                    [
                        PrimitiveGenerator(
                            "",
                            implied_qubits[float_ind],
                            -alpha * gamma / beta / 2 / (-sp.I),
                        )
                    ],
                    0,
                )
            elif implied_qubits[float_ind] != 0:
                raise ValueError("IDK WTF HAPPENED")
            # if it's 0 it's a purely complex number that washes out in phase
            implied_seq.pop(float_ind)
            implied_qubits.pop(float_ind)
            implied_value.pop(float_ind)
        else:
            raise NotImplementedError("Too many terms")
    beta = implied_value[0]

    new_value = (
        sp.sqrt(sp.I * alpha / beta) / (-sp.I)
    )  # extra division since automatically added in later, assumes value keeps Hermitian
    # print(f"{k=}")
    # print(f"{prim_AC=}")
    # print(f"{prim_BD=}")
    # print(f"{new_value=}")
    sequence_A = get_commutator_sequence(
        instrs[i][0], instrs, primitive_keys, -new_value
    )
    sequence_B = get_commutator_sequence(
        instrs[i][1], instrs, primitive_keys, -new_value
    )
    sequence_C = get_commutator_sequence(
        instrs[i][0], instrs, primitive_keys, new_value
    )
    sequence_D = get_commutator_sequence(
        instrs[i][1], instrs, primitive_keys, new_value
    )
    new_ops = sequence_A.ops + sequence_B.ops + sequence_C.ops + sequence_D.ops
    if extra_ops is not None:
        # print("extra ops", str(extra_ops.ops[0]))
        new_ops = extra_ops.ops + new_ops + extra_ops.ops

    return GateSequence(
        ops=new_ops,
        cost=new_value**3,  # TODO error bounding
    )


def sequence_summation(small: GateSequence, big: GateSequence):
    rescaled_small = GateSequence(
        [
            PrimitiveGenerator(*op.get_items()[:2], op.get_items()[2] / 2)
            for op in small.ops
        ],
        small.cost / 2,
    )
    return GateSequence(
        ops=rescaled_small.ops + big.ops + rescaled_small.ops, cost=big.cost
    )


def permutations(N):
    return [list(p) for p in itertools.permutations(range(N))]


def all_noncommutative_sums(options):
    return [
        sequence_summation_chain([options[i] for i in perm])
        for perm in permutations(len(options))
    ]


def sequence_summation_chain(seq):
    result = seq[0]
    for x in seq[1:]:
        result = sequence_summation(result, x)
    return result


def decompose_gensum(
    gensum: GeneratorSum, instrs: list, primitive_keys: List
) -> List[GateSequence]:
    generators, indices, values = gensum.lists()
    return [
        get_commutator_sequence(
            (generators[i], indices[i]),
            instrs=instrs,
            primitive_keys=primitive_keys,
            value=values[i],
        )
        for i in range(len(generators))
    ]


def gate_compile(genlist, instrs, primitive_keys) -> List[GateSequence]:
    # decompose all operations into commutation primitives
    sequence_lists = decompose_gensum(
        genlist,
        instrs=instrs,
        primitive_keys=primitive_keys,
    )  # List[GateSequence]
    print("<subsequences>")
    for seq in sequence_lists:
        seq.to_latex()
    print("</subsequences>")
    # print("Decomposed Gensums")
    # for seq_list in sequence_lists:
    #     for prim in seq_list.ops:
    #         prim.to_latex()
    #     seq_list.to_latex()
    print("Permutation count=", len(permutations(len(sequence_lists))))
    # find optimal way to sum them up
    return all_noncommutative_sums(sequence_lists)


def hamiltonian_to_gate(H, primitive_keys, optimal_decomps):
    keys = H.keys()
    ops = batch_forward_pass(
        keys, decomps=optimal_decomps, primitive_keys=primitive_keys
    )
    # print(f"{keys=}")
    # print(f"{ops=}")
    return gate_compile(H, instrs=ops, primitive_keys=primitive_keys)
