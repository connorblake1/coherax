import jax.numpy as jnp
import jax.random as jr
import numpy as np


def sqrt_psd(A):
    w, v = jnp.linalg.eigh((A + A.conj().T) * 0.5)
    w = jnp.clip(w, a_min=0.0)
    return (v * jnp.sqrt(w)) @ v.conj().T


class BinaryKrausTree:
    def __init__(self, K, leaf_assign):
        K = jnp.asarray(K)
        M0, N, _ = K.shape
        t = int(jnp.ceil(jnp.log2(M0)))
        Mpad = 1 << t
        if Mpad != M0:
            padK = jnp.zeros((Mpad - M0, N, N), dtype=K.dtype)
            K = jnp.concatenate([K, padK], axis=0)
            leaf_assign = jnp.asarray(leaf_assign).astype(int)
            if leaf_assign.size not in (M0, Mpad):
                raise ValueError(
                    "leaf_assign length must be M or the padded length 2^ceil(log2(M))."
                )
            if leaf_assign.size == M0:
                leaf_assign = jnp.concatenate(
                    [leaf_assign, jnp.arange(M0, Mpad, dtype=int)], axis=0
                )
        else:
            leaf_assign = jnp.asarray(leaf_assign).astype(int)
            if leaf_assign.size != M0:
                raise ValueError("leaf_assign length must equal M.")
        self.K = K
        self.M = self.K.conj().transpose(0, 2, 1) @ self.K
        self.leaf_assign = leaf_assign
        self.M_count, self.dim, _ = self.M.shape
        self.depth = int(jnp.log2(self.M_count))
        self._build()

    def _build(self):
        order = self.leaf_assign
        Ms = [self.M[order]]
        for level in range(self.depth, 0, -1):
            prev = Ms[-1]
            parent = (
                prev.reshape(2**level, self.dim, self.dim)
                .reshape(2 ** (level - 1), 2, self.dim, self.dim)
                .sum(axis=1)
            )
            Ms.append(parent)
        Ms = Ms[::-1]
        self.m_nodes = [
            [sqrt_psd(Ms[d][i]) for i in range(Ms[d].shape[0])] for d in range(len(Ms))
        ]
        B_nodes = []
        for d in range(1, len(self.m_nodes)):
            parent = self.m_nodes[d - 1]
            child = self.m_nodes[d]
            levelB = []
            for i in range(len(parent)):
                mp = parent[i]
                ma = child[2 * i]
                mb = child[2 * i + 1]
                mpinv = jnp.linalg.pinv(mp)
                ba = ma @ mpinv
                bb = mb @ mpinv
                levelB.append((ba, bb))
            B_nodes.append(levelB)
        self.B_nodes = B_nodes

    def effective_leaf_kraus(self, leaf_index):
        idx = int(leaf_index)
        bits = [(idx >> k) & 1 for k in range(self.depth - 1, -1, -1)]
        parent_index = 0
        meff = jnp.eye(self.dim, dtype=self.K.dtype)
        for d, bit in enumerate(bits):
            bpair = self.B_nodes[d][parent_index]
            meff = bpair[bit] @ meff
            parent_index = (parent_index << 1) | bit
        return meff

    def effective_leaf_effects(self):
        return jnp.stack(
            [
                self.effective_leaf_kraus(i).conj().T @ self.effective_leaf_kraus(i)
                for i in range(self.M_count)
            ],
            axis=0,
        )

    def check(self):
        E = self.effective_leaf_effects()
        D = E - self.M[self.leaf_assign]
        leaf_fro = jnp.linalg.norm(D, axis=(1, 2))
        diff = jnp.linalg.norm(leaf_fro)
        comp = jnp.linalg.norm(
            (E.sum(axis=0) - jnp.eye(self.dim, dtype=E.dtype)).reshape(-1)
        )
        return float(diff), float(comp)

    def _labels_for_level(self, l: int):
        return [("M_" + format(i, f"0{l}b")) for i in range(2**l)]

    def _level_effects_target(self):
        M_leaves = self.M[self.leaf_assign]  # (M,N,N), leaves in chosen order
        levels = []
        for l in range(1, self.depth + 1):
            block = 2 ** (self.depth - l)
            X = M_leaves.reshape(2**l, block, self.dim, self.dim).sum(axis=1)
            levels.append(X)
        return levels  # list of length depth; level l has shape (2**l, N, N)

    def _level_effects_synth(self):
        E = self.effective_leaf_effects()
        levels = []
        for l in range(1, self.depth + 1):
            block = 2 ** (self.depth - l)
            X = E.reshape(2**l, block, self.dim, self.dim).sum(axis=1)
            levels.append(X)
        return levels

    def visualize_tree(self, which="diff", precision=3, matrix=False):
        targ = self._level_effects_target()
        synth = self._level_effects_synth()
        for l in range(1, self.depth + 1):
            print(f"LEVEL {l - 1}:")
            labels = self._labels_for_level(l)
            T = targ[l - 1]
            S = synth[l - 1]
            if which == "target":
                blocks = T
            elif which == "synth":
                blocks = S
            elif which == "diff":
                for i in range(2**l):
                    d = jnp.linalg.norm((S[i] - T[i]).reshape(-1))
                    print(f"{labels[i]}: ||Δ||_F={float(d):.{precision}e}")
                continue
            else:
                raise ValueError("which ∈ {'target','synth','diff'}")
            for i in range(2**l):
                if matrix:
                    A = np.asarray(blocks[i])
                    Mstr = np.array2string(
                        np.round(A, precision),
                        precision=precision,
                        suppress_small=False,
                    )
                    print(f"{labels[i]} = {Mstr}")
                else:
                    tr = float(jnp.trace(blocks[i]).real)
                    nrm = float(jnp.linalg.norm(blocks[i].reshape(-1)))
                    print(
                        f"{labels[i]}: tr={tr:.{precision}f} ||·||_F={nrm:.{precision}f}"
                    )

    def measure_feedforward(self, rho, bits=None, key=None, return_intermediate=False):
        rho = jnp.asarray(rho)
        parent_index = 0
        bits_out = []
        labels_states = []
        probs = []
        label = ""
        for d in range(self.depth):
            b0, b1 = self.B_nodes[d][parent_index]
            x0 = b0 @ rho @ b0.conj().T
            x1 = b1 @ rho @ b1.conj().T
            p0 = jnp.real(jnp.trace(x0))
            p1 = jnp.real(jnp.trace(x1))
            if bits is not None:
                bit = int(bits[d])
            elif key is not None:
                key, sub = jr.split(key)
                bit = jr.bernoulli(sub, p1 / (p0 + p1 + 1e-15)).astype(int).item()
            else:
                bit = jnp.where(p1 > p0, 1, 0).item()
            x = jnp.where(bit == 0, x0, x1)
            p = jnp.where(bit == 0, p0, p1)
            rho = jnp.where(p > 0, x / p, rho)
            label = label + str(bit)
            labels_states.append(("M_" + label, rho))
            probs.append(p)
            bits_out.append(bit)
            parent_index = (parent_index << 1) | bit
        bitstring = "".join(str(b) for b in bits_out)
        if return_intermediate:
            return rho, bitstring, labels_states, jnp.stack(probs)
        return rho, bitstring
