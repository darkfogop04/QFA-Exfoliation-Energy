"""
Batched state-vector simulator for the QFA-Net quantum head, with exact gradients
by the adjoint method (one forward pass + one backward pass, no parameter shifts).

Circuit on n qubits, L layers (data re-uploading):
    for each layer l:
        RY(x_q)                         on every qubit q   (data, same angles each layer)
        RZ(a[l,q,0]) RY(a[l,q,1]) RZ(a[l,q,2])   on every qubit q (trainable, prunable)
        CNOT(q -> q+1 mod n)            ring
Readout: <Z_q> for every q and <Z_q Z_{q+1}> for every ring pair -> 2n numbers.

A pruned rotation (mask = False) is removed from the circuit (angle 0 = identity).
The last layer's final RZ commutes with every Z-type readout, so it can never change
the output; drop_dead() switches those n gates off from the start.

Qubit q is bit (n-1-q) of the basis-state index.
"""
from typing import Tuple

import numpy as np


class QuantumHead:
    def __init__(self, n_qubits: int = 8, n_layers: int = 3, rng: np.random.Generator = None,
                 init_scale: float = 0.1):
        rng = rng or np.random.default_rng(0)
        self.n, self.L = n_qubits, n_layers
        self.a = rng.normal(0.0, init_scale, size=(n_layers, n_qubits, 3))
        self.mask = np.ones((n_layers, n_qubits, 3), dtype=bool)
        dim = 2 ** n_qubits
        idx = np.arange(dim)
        bits = (idx[None, :] >> (n_qubits - 1 - np.arange(n_qubits))[:, None]) & 1  # (n, dim)
        z = 1.0 - 2.0 * bits
        pairs = [(q, (q + 1) % n_qubits) for q in range(n_qubits)] if n_qubits > 2 else [(0, 1)]
        self.pairs = pairs
        self.zsign = np.concatenate([z, np.stack([z[i] * z[j] for i, j in pairs])], axis=0)  # (2n, dim)
        self.cnot_perm = []
        for c, t in pairs:
            flip = idx ^ (1 << (n_qubits - 1 - t))
            self.cnot_perm.append(np.where(bits[c] == 1, flip, idx))

    # ------------------------------------------------------------------ helpers
    def drop_dead(self) -> None:
        self.mask[-1, :, 2] = False
        self.a[-1, :, 2] = 0.0

    @property
    def n_outputs(self) -> int:
        return self.zsign.shape[0]

    def n_active(self) -> int:
        return int(self.mask.sum())

    def gate_counts(self) -> dict:
        data = self.n * self.L
        cnot = len(self.pairs) * self.L
        return {"data_ry": data, "trainable": self.n_active(), "cnot": cnot,
                "total": data + self.n_active() + cnot}

    def _view(self, psi: np.ndarray, q: int) -> np.ndarray:
        B = psi.shape[0]
        return psi.reshape(B, 2 ** q, 2, 2 ** (self.n - q - 1))

    @staticmethod
    def _ry(v: np.ndarray, theta) -> None:
        c = np.cos(np.asarray(theta) / 2.0)
        s = np.sin(np.asarray(theta) / 2.0)
        if np.ndim(c):
            c = c[:, None, None]
            s = s[:, None, None]
        a0 = v[:, :, 0, :].copy()
        a1 = v[:, :, 1, :]
        v[:, :, 0, :] = c * a0 - s * a1
        v[:, :, 1, :] = s * a0 + c * a1

    @staticmethod
    def _rz(v: np.ndarray, theta) -> None:
        ph = np.exp(-0.5j * np.asarray(theta))
        if np.ndim(ph):
            ph = ph[:, None, None]
        v[:, :, 0, :] *= ph
        v[:, :, 1, :] *= np.conj(ph)

    def _ops(self):
        """Gate sequence as (kind, qubit, layer, slot) with kind in {'x','ry','rz','cnot'}."""
        ops = []
        for l in range(self.L):
            for q in range(self.n):
                ops.append(("x", q, l, -1))
            for q in range(self.n):
                for k, kind in enumerate(("rz", "ry", "rz")):
                    if self.mask[l, q, k]:
                        ops.append((kind, q, l, k))
            for p in range(len(self.pairs)):
                ops.append(("cnot", p, l, -1))
        return ops

    def _apply(self, psi, op, x, sign: float = 1.0):
        kind, q, l, k = op
        if kind == "cnot":
            return psi[:, self.cnot_perm[q]]
        v = self._view(psi, q)
        if kind == "x":
            self._ry(v, sign * x[:, q])
        elif kind == "ry":
            self._ry(v, sign * self.a[l, q, k])
        else:
            self._rz(v, sign * self.a[l, q, k])
        return psi

    # ------------------------------------------------------------------ forward / backward
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """x: (B, n) angles. Returns (outputs (B, 2n), final state)."""
        B = x.shape[0]
        psi = np.zeros((B, 2 ** self.n), dtype=complex)
        psi[:, 0] = 1.0
        for op in self._ops():
            psi = self._apply(psi, op, x)
        probs = (psi.real ** 2 + psi.imag ** 2)
        return probs @ self.zsign.T, psi

    def backward(self, x: np.ndarray, psi: np.ndarray, g: np.ndarray):
        """g = dL/d(outputs) (B, 2n). Returns (dL/da (L,n,3), dL/dx (B,n))."""
        lam = psi * (g @ self.zsign)
        psi = psi.copy()
        da = np.zeros_like(self.a)
        dx = np.zeros_like(x)
        for op in reversed(self._ops()):
            kind, q, l, k = op
            if kind != "cnot":
                v = self._view(psi, q)
                w = self._view(lam, q)
                if kind in ("x", "ry"):   # Im <lam| Y psi>
                    inner = (np.conj(w[:, :, 0, :]) * (-1j) * v[:, :, 1, :]
                             + np.conj(w[:, :, 1, :]) * (1j) * v[:, :, 0, :]).sum(axis=(1, 2))
                else:                     # Im <lam| Z psi>
                    inner = (np.conj(w[:, :, 0, :]) * v[:, :, 0, :]
                             - np.conj(w[:, :, 1, :]) * v[:, :, 1, :]).sum(axis=(1, 2))
                grad = inner.imag
                if kind == "x":
                    dx[:, q] += grad
                else:
                    da[l, q, k] += grad.sum()
            psi = self._apply(psi, op, x, sign=-1.0)
            lam = self._apply(lam, op, x, sign=-1.0)
        da[~self.mask] = 0.0
        return da, dx
