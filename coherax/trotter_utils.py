import sympy as sp
import re
from typing import List, Any
from itertools import groupby
from sympy import Expr, Integer, Float, Mul
from IPython.display import display, Math, Latex

I = sp.I


## PRINTING UTILS
def chop_floats(s: str, n: int = 3) -> str:
    # rounds off floats interpreted from str
    float_re = re.compile(r"\d+\.\d+")

    def repl(m):
        i, _, f = m.group(0).partition(".")
        return i + "." + f[:n]

    return float_re.sub(repl, s)


def latex(expr):
    # displays within ipynb
    if isinstance(expr, Expr):
        display(Math(sp.latex(expr)))
    elif isinstance(expr, str):
        display(Latex("$" + chop_floats(expr).replace("*I", "i") + "$"))
    else:
        raise TypeError(f"Encountered {type(expr)}")


def collapse_to_latex(s: str) -> str:
    # ie qpqppq -> qpqp^2q
    segments = [(char, sum(1 for _ in group)) for char, group in groupby(s)]

    return "".join(
        f"{char}^{{{count}}}" if count > 1 else char for char, count in segments
    )


## OPERATOR UTILS
def pauli_product(i, j):
    if i == 0:
        return j, 1
    elif j == 0:
        return i, 1
    elif i == j:
        return 0, 1
    eps = {(1, 2): 3, (2, 3): 1, (3, 1): 2}
    if (i, j) in eps:
        return eps[(i, j)], I
    k = eps[(j, i)]
    return k, -I


pauli_from_index = {1: "x", 2: "y", 3: "z"}


class GeneratorSum:
    qp_sequences: List[str]
    qubit_indices: List[int]
    values: List[Any]

    def __init__(self, qp_sequences, qubit_indices, values=1):
        if (
            isinstance(qp_sequences, str)
            and isinstance(qubit_indices, int)
            and isinstance(values, (Expr, int, float))
        ):
            self.qp_sequences = [qp_sequences]
            self.qubit_indices = [qubit_indices]
            self.values = [sp.S(values)]
        elif (
            isinstance(qp_sequences, List)
            and isinstance(qubit_indices, List)
            and isinstance(values, List)
        ):
            if len(qp_sequences) != len(qubit_indices) or len(qp_sequences) != len(
                values
            ):
                raise ValueError(
                    f"Mismatched sequence lengths: {len(qp_sequences)},{len(qubit_indices)},{len(values)}"
                )
            self.qp_sequences = qp_sequences
            self.qubit_indices = qubit_indices
            self.values = values
        else:
            raise TypeError(
                "Types:", type(qp_sequences), type(qubit_indices), type(values)
            )

    def __mul__(self, other):
        if isinstance(other, (complex, float, int)):
            return GeneratorSum(
                self.qp_sequences,
                self.qubit_indices,
                [other * value for value in self.values],
            )
        else:
            raise TypeError

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, GeneratorSum):
            all_seqs = self.qp_sequences + other.qp_sequences
            all_indices = self.qubit_indices + other.qubit_indices
            all_values = self.values + other.values
            d = {}
            for seq, index, value in zip(all_seqs, all_indices, all_values):
                d[(seq, index)] = d.get((seq, index), 0) + value
            new_data = [(k, d[k]) for k in d if (d[k] != 0)]
            return GeneratorSum(
                [item[0][0] for item in new_data],
                [item[0][1] for item in new_data],
                [item[1] for item in new_data],
            )
        else:
            raise TypeError

    def canonicalize(self, squeezing_allowed=False):
        if squeezing_allowed:
            raise NotImplementedError

        def _canon(key):
            s = key[0]
            ind = key[1]
            if "pq" not in s:
                return {key: 1}
            i = s.index("pq")
            x, y = s[:i], s[i + 2 :]
            m1 = _canon((x + "qp" + y, ind))
            m2 = _canon((x + y, ind))
            d = {}
            for k, v in m1.items():
                d[k] = d.get(k, 0) + v
            for k, v in m2.items():
                d[k] = d.get(k, 0) - I * v
            return d

        terms = []
        for seq, index, value in zip(
            self.qp_sequences, self.qubit_indices, self.values
        ):
            terms.append((_canon((seq, index)), value))

        # return sum([GeneratorSum(k[0],k[1],value*v) for term, value in terms for k,v in term.items()])
        new_item = GeneratorSum("", 0, 0)
        for term, value in terms:
            for k, v in term.items():
                if v == 0:
                    continue
                new_item = new_item + GeneratorSum(k[0], k[1], value * v)

        return new_item

    def pretty(self):
        out_str = ""
        for seq, index, value in zip(
            self.qp_sequences, self.qubit_indices, self.values
        ):
            if value != 1:
                if isinstance(value, (Integer, Float)):
                    out_str = out_str + str(value) + " "
                else:
                    out_str = out_str + value._repr_latex_().strip("$") + " "
            if seq != "":
                out_str = out_str + collapse_to_latex(seq) + " "
            if index != 0:
                out_str = out_str + "\sigma_" + str(pauli_from_index[index]) + " "
            out_str = out_str + "+ "
        return out_str.rstrip(" +")

    def __str__(self):
        return self.pretty()

    def to_latex(self):
        latex(str(self))

    def commutator(A, B):
        if isinstance(A, GeneratorSum) and isinstance(B, GeneratorSum):
            d = {}
            for s1, i1, v1 in zip(A.qp_sequences, A.qubit_indices, A.values):
                for s2, i2, v2 in zip(B.qp_sequences, B.qubit_indices, B.values):
                    i3, factor3 = pauli_product(i1, i2)
                    d[(s1 + s2, i3)] = d.get((s1 + s2, i3), 0) + v1 * v2 * factor3
                    i4, factor4 = pauli_product(i2, i1)
                    d[(s2 + s1, i3)] = d.get((s2 + s1, i4), 0) - v1 * v2 * factor4
            new_data = [(k, d[k]) for k in sorted(d) if (d[k] != 0 or k == ("", 0))]
            return GeneratorSum(
                [item[0][0] for item in new_data],
                [item[0][1] for item in new_data],
                [item[1] for item in new_data],
            )
        else:
            raise TypeError

    def lists(self):
        return self.qp_sequences, self.qubit_indices, self.values

    def keys(self):
        return [(k, i) for k, i in zip(self.qp_sequences, self.qubit_indices)]

    def concretize(self, swap_dict: dict):
        concrete_values = []
        for val in self.values:
            if not isinstance(val, (int, float, complex)):
                concrete_values.append(val.evalf(subs=swap_dict))
            else:
                concrete_values.append(val)
        return GeneratorSum(self.qp_sequences, self.qubit_indices, concrete_values)


class PrimitiveGenerator:
    seq: str
    ind: int
    val: Any

    def __init__(self, seq, ind, val):
        if not (
            isinstance(seq, str)
            and isinstance(ind, int)
            and isinstance(val, (Expr, Mul, int, float, complex))
        ):
            raise ValueError(
                f"Invalid Primitive Generator Types: {type(seq)=}{type(ind)=}{type(val)=}"
            )
        self.seq = seq
        self.ind = ind
        self.val = val

    def pretty(self):
        out_str = ""
        if self.val != 1:
            if isinstance(self.val, (Integer, Float)):
                out_str = out_str + str(self.val) + " "
            else:
                self.val = sp.simplify(self.val)
                out_str = out_str + self.val._repr_latex_().strip("$") + " "
        if self.seq != "":
            out_str = out_str + collapse_to_latex(self.seq) + " "
        if self.ind != 0:
            out_str = out_str + "\sigma_" + str(pauli_from_index[self.ind]) + " "
        return out_str

    def __str__(self):
        return self.pretty()

    def to_latex(self):
        latex(str(self))

    def get_items(self):
        return (self.seq, self.ind, self.val)


class GateSequence:
    ops: List[PrimitiveGenerator]
    cost: float

    def __init__(self, ops: List[PrimitiveGenerator], cost: float):
        self.ops = ops
        self.cost = cost

    def get_sequence(self):
        return self.operations

    def get_cost(self):
        return self.cost

    def __str__(self):
        out_str = (
            r"\text{GateSequence(Sequence}="
            + "".join(
                [
                    r"\exp\left(-i\left(" + str(prim) + r"\right)\right)"
                    for prim in self.ops
                ]
            )
            + r", \text{Error}="
        )
        if isinstance(self.cost, (int, float, complex)):
            out_str = out_str + f"{self.cost:.2f}"
        else:
            out_str = out_str + self.cost._repr_latex_().strip("$")
        out_str = out_str + ")"
        return out_str

    def to_latex(self):
        latex(str(self))
