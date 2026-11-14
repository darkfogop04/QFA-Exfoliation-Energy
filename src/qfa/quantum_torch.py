"""
PyTorch version of the QFA-Net quantum parts (autograd does the gradients).

QuantumHeadTorch mirrors src/qfa/lab/qsim.py gate by gate (tests/test_qfa.py checks that
both give the same outputs), and QTGeneratorTorch mirrors src/qfa/lab/qt.py.
"""
import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn


class QuantumHeadTorch(nn.Module):
    """n qubits, L re-uploading layers: RY(x) | RZ RY RZ (trainable, maskable) | CNOT ring.
    Readout <Z_q> and <Z_q Z_q+1>: 2n numbers."""

    def __init__(self, n_qubits: int = 8, n_layers: int = 3, init_scale: float = 0.1):
        super().__init__()
        self.n, self.L = n_qubits, n_layers
        self.a = nn.Parameter(init_scale * torch.randn(n_layers, n_qubits, 3))
        self.register_buffer("mask", torch.ones(n_layers, n_qubits, 3, dtype=torch.bool))
        dim = 2 ** n_qubits
        idx = torch.arange(dim)
        bits = (idx[None, :] >> (n_qubits - 1 - torch.arange(n_qubits))[:, None]) & 1
        z = 1.0 - 2.0 * bits.float()
        pairs = [(q, (q + 1) % n_qubits) for q in range(n_qubits)] if n_qubits > 2 else [(0, 1)]
        self.pairs = pairs
        zs = torch.cat([z, torch.stack([z[i] * z[j] for i, j in pairs])], dim=0)
        self.register_buffer("zsign", zs)
        perms = []
        for c, t in pairs:
            flip = idx ^ (1 << (n_qubits - 1 - t))
            perms.append(torch.where(bits[c] == 1, flip, idx))
        self.register_buffer("perms", torch.stack(perms))

    def drop_dead(self) -> None:
        with torch.no_grad():
            self.mask[-1, :, 2] = False
            self.a[-1, :, 2] = 0.0

    @property
    def n_outputs(self) -> int:
        return self.zsign.shape[0]

    def n_active(self) -> int:
        return int(self.mask.sum())

    def gate_counts(self) -> Dict[str, int]:
        data = self.n * self.L
        cnot = len(self.pairs) * self.L
        return {"data_ry": data, "trainable": self.n_active(), "cnot": cnot,
                "total": data + self.n_active() + cnot}

    def _ry(self, psi, q, theta):
        B = psi.shape[0]
        v = psi.reshape(B, 2 ** q, 2, 2 ** (self.n - q - 1))
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        if c.dim() == 1:
            c, s = c[:, None, None], s[:, None, None]
        a0, a1 = v[:, :, 0, :], v[:, :, 1, :]
        return torch.stack([c * a0 - s * a1, s * a0 + c * a1], dim=2).reshape(B, -1)

    def _rz(self, psi, q, theta):
        B = psi.shape[0]
        v = psi.reshape(B, 2 ** q, 2, 2 ** (self.n - q - 1))
        ph = torch.exp(-0.5j * theta.to(psi.dtype))
        return torch.stack([v[:, :, 0, :] * ph, v[:, :, 1, :] * torch.conj(ph)], dim=2).reshape(B, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n) angles -> (B, 2n) expectation values."""
        B = x.shape[0]
        cdt = torch.complex128 if x.dtype == torch.float64 else torch.complex64
        psi = torch.zeros(B, 2 ** self.n, dtype=cdt, device=x.device)
        psi[:, 0] = 1.0
        a = self.a * self.mask
        for l in range(self.L):
            for q in range(self.n):
                psi = self._ry(psi, q, x[:, q])
            for q in range(self.n):
                if self.mask[l, q, 0]:
                    psi = self._rz(psi, q, a[l, q, 0])
                if self.mask[l, q, 1]:
                    psi = self._ry(psi, q, a[l, q, 1])
                if self.mask[l, q, 2]:
                    psi = self._rz(psi, q, a[l, q, 2])
            for p in range(len(self.pairs)):
                psi = psi[:, self.perms[p]]
        probs = psi.real ** 2 + psi.imag ** 2
        return probs.to(x.dtype) @ self.zsign.t().to(x.dtype)


class QTGeneratorTorch(nn.Module):
    """Quantum-Train: n = ceil(log2 P) qubits, L layers of RY + CNOT chain, shared mapping MLP.
    generate() returns the P classical weights as one flat vector."""

    def __init__(self, shapes: Sequence[Tuple[int, ...]], n_layers: int = 8, hidden: int = 16,
                 init_std: Sequence[float] = None):
        super().__init__()
        self.shapes = [tuple(s) for s in shapes]
        self.sizes = [int(math.prod(s)) for s in self.shapes]
        self.P = sum(self.sizes)
        self.n = max(1, math.ceil(math.log2(self.P)))
        self.L = n_layers
        self.phi = nn.Parameter(2 * math.pi * torch.rand(n_layers, self.n))
        self.mapper = nn.Sequential(nn.Linear(self.n + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                    nn.SiLU(), nn.Linear(hidden, 1))
        scales = init_std or [math.sqrt(2.0 / s[1]) if len(s) > 1 else 0.05 for s in self.shapes]
        self.scales = nn.Parameter(torch.tensor(scales, dtype=torch.float32))
        dim = 2 ** self.n
        idx = torch.arange(dim)
        bits = ((idx[:, None] >> (self.n - 1 - torch.arange(self.n))[None, :]) & 1).float()
        self.register_buffer("bits_pm", 2 * bits[: self.P] - 1)
        self.register_buffer("tensor_id", torch.repeat_interleave(torch.arange(len(self.sizes)),
                                                                  torch.tensor(self.sizes)))
        chain = []
        for q in range(self.n - 1):
            flip = idx ^ (1 << (self.n - 1 - (q + 1)))
            ctrl = (idx >> (self.n - 1 - q)) & 1
            chain.append(torch.where(ctrl == 1, flip, idx))
        self.register_buffer("chain", torch.stack(chain) if chain else torch.zeros(0, dim, dtype=torch.long))

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def state(self) -> torch.Tensor:
        psi = torch.zeros(2 ** self.n, device=self.phi.device)
        psi[0] = 1.0
        for l in range(self.L):
            for q in range(self.n):
                v = psi.reshape(2 ** q, 2, 2 ** (self.n - q - 1))
                c, s = torch.cos(self.phi[l, q] / 2), torch.sin(self.phi[l, q] / 2)
                psi = torch.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], dim=1).reshape(-1)
            for perm in self.chain:
                psi = psi[perm]
        return psi

    def generate(self) -> torch.Tensor:
        p = self.state()[: self.P] ** 2
        feats = torch.cat([self.bits_pm, (2 ** self.n * p - 1.0)[:, None]], dim=1)
        m = self.mapper(feats)[:, 0]
        return self.scales[self.tensor_id] * m

    def as_dict(self, names: List[str]) -> Dict[str, torch.Tensor]:
        flat = self.generate()
        out, k = {}, 0
        for name, s, n in zip(names, self.shapes, self.sizes):
            out[name] = flat[k:k + n].reshape(s)
            k += n
        return out
