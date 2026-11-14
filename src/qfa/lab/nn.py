"""
A small NumPy MLP with hand-written backpropagation and an AdamW optimiser.

Used by the lab experiments (this machine has no PyTorch). Everything is float64.
Parameters are kept as a list of arrays so that Quantum-Train can overwrite them
from a flat vector (set_flat) and read their gradients as one flat vector.
"""
from typing import List, Optional, Sequence

import numpy as np


def silu(x):
    return x / (1.0 + np.exp(-x))


def silu_grad(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s * (1.0 + x * (1.0 - s))


class MLP:
    """Linear -> SiLU -> ... -> Linear. Last layer has no activation."""

    def __init__(self, sizes: Sequence[int], rng: np.random.Generator, last_zero: bool = False):
        self.sizes = list(sizes)
        self.params: List[np.ndarray] = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            W = rng.normal(0.0, np.sqrt(2.0 / a), size=(a, b))
            if last_zero and i == len(sizes) - 2:
                W = np.zeros((a, b))
            self.params += [W, np.zeros(b)]
        self._cache = None

    @property
    def n_params(self) -> int:
        return int(sum(p.size for p in self.params))

    def forward(self, x: np.ndarray, keep: bool = True):
        """Returns (output, last hidden activation)."""
        acts = [x]
        pres = []
        h = x
        n_layers = len(self.params) // 2
        for i in range(n_layers):
            W, b = self.params[2 * i], self.params[2 * i + 1]
            z = h @ W + b
            if i < n_layers - 1:
                pres.append(z)
                h = silu(z)
                acts.append(h)
            else:
                h = z
        if keep:
            self._cache = (acts, pres)
        return h, acts[-1]

    def backward(self, gout: np.ndarray, ghidden: Optional[np.ndarray] = None):
        """gout: dL/d(output). ghidden: optional extra dL/d(last hidden activation).
        Returns (list of param grads, dL/dx)."""
        acts, pres = self._cache
        n_layers = len(self.params) // 2
        grads = [None] * len(self.params)
        g = gout
        for i in reversed(range(n_layers)):
            W = self.params[2 * i]
            a_in = acts[i]
            grads[2 * i] = a_in.T @ g
            grads[2 * i + 1] = g.sum(axis=0)
            g = g @ W.T
            if i > 0:
                if i == n_layers - 1 and ghidden is not None:
                    g = g + ghidden
                g = g * silu_grad(pres[i - 1])
        return grads, g

    def get_flat(self) -> np.ndarray:
        return np.concatenate([p.ravel() for p in self.params])

    def set_flat(self, v: np.ndarray) -> None:
        k = 0
        for i, p in enumerate(self.params):
            self.params[i] = v[k:k + p.size].reshape(p.shape).copy()
            k += p.size


class AdamW:
    def __init__(self, params: List[np.ndarray], lr: float = 1e-3, wd: float = 0.0,
                 betas=(0.9, 0.999), eps: float = 1e-8, decay_mask: Optional[List[bool]] = None):
        self.lr, self.wd, self.b1, self.b2, self.eps = lr, wd, betas[0], betas[1], eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
        self.decay_mask = decay_mask or [p.ndim > 1 for p in params]

    def step(self, params: List[np.ndarray], grads: List[np.ndarray]) -> None:
        self.t += 1
        c1 = 1 - self.b1 ** self.t
        c2 = 1 - self.b2 ** self.t
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            upd = (self.m[i] / c1) / (np.sqrt(self.v[i] / c2) + self.eps)
            if self.wd and self.decay_mask[i]:
                p -= self.lr * self.wd * p
            p -= self.lr * upd


def loss_and_grad(pred: np.ndarray, y: np.ndarray, task_type: str):
    """pred, y: shape (B,). Regression: L1 on standardised target. Classification: BCE with logits."""
    if task_type == "regression":
        d = pred - y
        return float(np.mean(np.abs(d))), np.sign(d) / len(y)
    p = 1.0 / (1.0 + np.exp(-pred))
    eps = 1e-12
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    return float(loss), (p - y) / len(y)
