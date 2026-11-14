"""
Quantum-Train (Liu et al., 2024) in NumPy, with exact gradients.

A classical network with P weights is NOT trained directly. Instead:
    n = ceil(log2 P) qubits, circuit of L layers [RY(phi) on every qubit, CNOT chain]
    |psi(phi)>  ->  probabilities p_k = |<k|psi>|^2  for k = 0 .. 2^n - 1
    weight k    =  scale_t * M_beta( bits(k), 2^n * p_k - 1 )        for k < P
M_beta is a small shared MLP (the "mapping model"); scale_t is one number per weight tensor.
Trained parameters: phi (n*L) + beta + scales  -> O(polylog P) instead of P.
After training the weights are copied into the ordinary network ("materialise"), so
prediction needs no quantum computer.

The RY/CNOT circuit has real amplitudes; gradients use the adjoint method.
"""
from typing import List, Sequence

import numpy as np

from .nn import MLP


class QTGenerator:
    def __init__(self, shapes: Sequence[tuple], rng: np.random.Generator, n_layers: int = 8,
                 hidden: int = 16, mode: str = "quantum"):
        """mode='none' is the classical control: the mapping MLP sees only the bit-string of k
        (its probability input is fixed at 0), so the quantum circuit plays no part."""
        self.mode = mode
        self.shapes = [tuple(s) for s in shapes]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.P = int(sum(self.sizes))
        self.n = max(1, int(np.ceil(np.log2(self.P))))
        self.L = n_layers
        self.phi = rng.uniform(0.0, 2 * np.pi, size=(n_layers, self.n))
        self.mapper = MLP([self.n + 1, hidden, hidden, 1], rng)
        # one scale per tensor, start at the usual He-init size for weights, small for biases
        self.scales = np.array([np.sqrt(2.0 / s[0]) if len(s) > 1 else 0.05 for s in self.shapes])
        dim = 2 ** self.n
        idx = np.arange(dim)
        self.bits = ((idx[:, None] >> (self.n - 1 - np.arange(self.n))[None, :]) & 1).astype(float)
        self.inputs_bits = 2.0 * self.bits[: self.P] - 1.0
        self.tensor_id = np.repeat(np.arange(len(self.sizes)), self.sizes)
        self.chain = []
        for q in range(self.n - 1):
            flip = idx ^ (1 << (self.n - 1 - (q + 1)))
            ctrl = (idx >> (self.n - 1 - q)) & 1
            self.chain.append(np.where(ctrl == 1, flip, idx))

    @property
    def n_trainable(self) -> int:
        q = self.phi.size if self.mode == "quantum" else 0
        return int(q + self.mapper.n_params + self.scales.size)

    def _ry(self, psi, q, theta):
        v = psi.reshape(2 ** q, 2, 2 ** (self.n - q - 1))
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        a0 = v[:, 0, :].copy()
        a1 = v[:, 1, :]
        v[:, 0, :] = c * a0 - s * a1
        v[:, 1, :] = s * a0 + c * a1

    def _state(self):
        psi = np.zeros(2 ** self.n)
        psi[0] = 1.0
        for l in range(self.L):
            for q in range(self.n):
                self._ry(psi, q, self.phi[l, q])
            for perm in self.chain:
                psi = psi[perm]
        return psi

    def generate(self):
        """Returns flat weight vector (P,). Keeps what backward needs."""
        psi = self._state()
        p = psi[: self.P] ** 2 if self.mode == "quantum" else np.full(self.P, 2.0 ** -self.n)
        feats = np.concatenate([self.inputs_bits, (2 ** self.n * p - 1.0)[:, None]], axis=1)
        m, _ = self.mapper.forward(feats)
        m = m[:, 0]
        self._cache = (psi, m)
        return self.scales[self.tensor_id] * m

    def backward(self, gw: np.ndarray):
        """gw = dL/d(weights) (P,). Returns grads for (phi, mapper params, scales)."""
        psi, m = self._cache
        g_scales = np.bincount(self.tensor_id, weights=gw * m, minlength=len(self.sizes))
        gm = (self.scales[self.tensor_id] * gw)[:, None]
        g_map, g_in = self.mapper.backward(gm)
        gp = np.zeros(2 ** self.n)
        gp[: self.P] = (2 ** self.n) * g_in[:, -1]
        # L depends on psi through p_k = psi_k^2  ->  dL/dpsi_k = 2 psi_k gp_k
        if self.mode != "quantum":
            return np.zeros_like(self.phi), g_map, g_scales
        lam = 2.0 * psi * gp
        psi = psi.copy()
        g_phi = np.zeros_like(self.phi)
        for l in reversed(range(self.L)):
            for perm in reversed(self.chain):
                inv = np.argsort(perm)
                psi = psi[inv]
                lam = lam[inv]
            for q in reversed(range(self.n)):
                v = psi.reshape(2 ** q, 2, 2 ** (self.n - q - 1))
                w = lam.reshape(2 ** q, 2, 2 ** (self.n - q - 1))
                # d psi / d theta = -1/2 * iY psi ; for real states: (-1/2)[[0,-1],[1,0]] psi
                g_phi[l, q] = 0.5 * float((w[:, 0, :] * (-v[:, 1, :]) + w[:, 1, :] * v[:, 0, :]).sum())
                self._ry(psi, q, -self.phi[l, q])
                self._ry(lam, q, -self.phi[l, q])
        return g_phi, g_map, g_scales

    def split(self, flat: np.ndarray) -> List[np.ndarray]:
        out, k = [], 0
        for s, n in zip(self.shapes, self.sizes):
            out.append(flat[k:k + n].reshape(s))
            k += n
        return out
