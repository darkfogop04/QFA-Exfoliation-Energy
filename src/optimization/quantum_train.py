"""
Quantum-Train (Liu et al., arXiv:2405.11304) for the HQ-GNN.

A classical network with M weights is not trained directly. Instead:

    n = ceil(log2 M) qubits, QNN with trainable angles phi
    measure  -> 2^n probabilities p(k), keep the first M
    mapping model G_gamma([bits(k), p(k)]) -> weight w_k      (a small MLP)
    run the classical network with the generated weights
    loss.backward() updates only phi and gamma

The circuit runs ONCE per optimisation step (not once per crystal), so its
cost does not grow with batch size. After training, `materialize()` writes the
generated weights into the classical network and the QNN is dropped:
inference is purely classical.

What it does and does not save
  + trainable parameters: ~72.7k  ->  ~2k  (QNN + mapping + small extras)
  + optimiser state, and (the point for jdft2d) a much smaller hypothesis
    space -> less over-fitting on 509 training crystals
  - the inference network is the SAME size as before; QT is a training-time
    reparameterisation, not a smaller model
  - training adds one 17-qubit state-vector simulation per step

Practical additions on top of the paper (each can be switched off):
  * BatchNorm / LayerNorm parameters are never generated (384 values that
    must start at 1 / 0); they stay ordinary trainable parameters.
  * Per-tensor standardisation + learnable (scale, shift): every weight
    tensor gets the right magnitude at initialisation, regardless of the
    mapping model's raw output range.
  * mode="delta": w = w_init (frozen) + generated. Usually trains far more
    stably than generating the full weights ("full", the paper's setting).

Controls for a fair quantum-vs-classical comparison (generator=...):
  "quantum"   - the method above
  "classical" - same budget: p = softmax(R z), z has as many entries as the
                QNN has angles, R is a fixed random matrix
  "none"      - p(k) set to 0: the mapping MLP alone maps bit-strings to
                weights. If this matches "quantum", the QNN is not what helps.
"""
import math
from typing import Dict, List, Sequence

import pennylane as qml
import torch
import torch.nn as nn
from torch.func import functional_call

_NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)


class QTGenerator(nn.Module):
    def __init__(self, n_weights: int, n_qnn_layers: int = 8, mapping_hidden: int = 32,
                 generator: str = "quantum", diff_method: str = "backprop"):
        super().__init__()
        if generator not in ("quantum", "classical", "none"):
            raise ValueError(f"generator must be quantum|classical|none, got {generator!r}")
        self.n_weights = n_weights
        self.n_qubits = n = max(1, math.ceil(math.log2(max(2, n_weights))))
        self.generator = generator

        # Basis-state labels as +-1, wire 0 = most significant bit (PennyLane order).
        idx = torch.arange(n_weights).unsqueeze(1)
        shifts = torch.arange(n - 1, -1, -1).unsqueeze(0)
        self.register_buffer("bits", ((idx >> shifts) & 1).float() * 2 - 1)

        if generator == "quantum":
            self.qnn_weights = nn.Parameter(0.1 * torch.randn(n_qnn_layers, n))
            dev = qml.device("default.qubit", wires=n)

            @qml.qnode(dev, interface="torch", diff_method=diff_method)
            def qnn(w):
                for i in range(n):
                    qml.Hadamard(wires=i)          # start from the uniform distribution
                for l in range(w.shape[0]):
                    for i in range(n):
                        qml.RY(w[l, i], wires=i)
                    for i in range(n - 1):
                        qml.CNOT(wires=[i, i + 1])
                return qml.probs(wires=list(range(n)))

            self.qnn = qnn
        elif generator == "classical":
            k = n_qnn_layers * n
            self.z = nn.Parameter(0.1 * torch.randn(k))
            self.register_buffer("proj", torch.randn(2 ** n, k) / math.sqrt(k))

        self.mapping = nn.Sequential(
            nn.Linear(n + 1, mapping_hidden), nn.ReLU(),
            nn.Linear(mapping_hidden, mapping_hidden), nn.ReLU(),
            nn.Linear(mapping_hidden, 1),
        )

    def probs(self) -> torch.Tensor:
        if self.generator == "quantum":
            p = self.qnn(self.qnn_weights)
        elif self.generator == "classical":
            p = torch.softmax(self.proj @ self.z, dim=0)
        else:
            p = torch.full((2 ** self.n_qubits,), 2.0 ** -self.n_qubits)
        return p.to(self.bits.dtype).to(self.bits.device)

    def forward(self) -> torch.Tensor:
        p = self.probs()[: self.n_weights]
        p_feat = p * (2 ** self.n_qubits) - 1.0       # 0 for a uniform distribution
        feats = torch.cat([self.bits, p_feat.unsqueeze(1)], dim=1)
        return self.mapping(feats).squeeze(1)         # (M,)


class QuantumTrainModel(nn.Module):
    """Wraps a base model; the selected classical weights are generated."""

    def __init__(self, base: nn.Module, qt_cfg: Dict):
        super().__init__()
        self.base = base
        targets: Sequence[str] = qt_cfg.get("targets", ["gnn", "mlp"])
        self.mode = qt_cfg.get("mode", "full")
        self.standardize = qt_cfg.get("standardize", True)
        if self.mode not in ("full", "delta"):
            raise ValueError(f"mode must be full|delta, got {self.mode!r}")

        names: List[str] = []
        shapes, numels, init_std, init_flat = [], [], [], []
        for mod_name, mod in base.named_modules():
            if isinstance(mod, _NORM_TYPES):
                continue
            for p_name, p in mod.named_parameters(recurse=False):
                full = f"{mod_name}.{p_name}" if mod_name else p_name
                if full.startswith("quantum."):
                    continue  # circuit angles: trained / pruned directly
                if not any(full == t or full.startswith(t + ".") for t in targets):
                    continue
                names.append(full)
                shapes.append(tuple(p.shape))
                numels.append(p.numel())
                init_std.append(max(p.detach().std().item() if p.numel() > 1 else 0.0, 1e-3))
                init_flat.append(p.detach().flatten().clone())
                p.requires_grad_(False)
        if not names:
            raise ValueError(f"No parameters matched QT targets {targets}")

        self.names, self.shapes, self.numels = names, shapes, numels
        self.n_generated = sum(numels)

        std = torch.tensor(init_std)
        self.scale = nn.Parameter(std * (0.1 if self.mode == "delta" else 1.0))
        self.shift = nn.Parameter(torch.zeros(len(names)))
        if self.mode == "delta":
            self.register_buffer("w0", torch.cat(init_flat))

        self.generator = QTGenerator(
            self.n_generated,
            n_qnn_layers=qt_cfg.get("n_qnn_layers", 8),
            mapping_hidden=qt_cfg.get("mapping_hidden", 32),
            generator=qt_cfg.get("generator", "quantum"),
            diff_method=qt_cfg.get("diff_method", "backprop"),
        )
        self._cache = None

    # ---------------------------------------------------------- weights
    def generate(self) -> Dict[str, torch.Tensor]:
        g = self.generator()
        chunks = torch.split(g, self.numels)
        w0_chunks = torch.split(self.w0, self.numels) if self.mode == "delta" else None
        params = {}
        for t, (name, shape, c) in enumerate(zip(self.names, self.shapes, chunks)):
            if self.standardize and c.numel() > 1:
                c = (c - c.mean()) / (c.std(unbiased=False) + 1e-6)
            w = self.shift[t] + self.scale[t] * c
            if w0_chunks is not None:
                w = w0_chunks[t] + w
            params[name] = w.view(shape)
        return params

    def train(self, mode: bool = True):
        self._cache = None
        return super().train(mode)

    def forward(self, data):
        if not self.training and self._cache is not None:
            params = self._cache
        else:
            params = self.generate()
            if not self.training:
                self._cache = {k: v.detach() for k, v in params.items()}
        return functional_call(self.base, params, (data,))

    # ------------------------------------------------------- reporting
    def trainable_breakdown(self) -> Dict[str, int]:
        gen = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        aff = self.scale.numel() + self.shift.numel()
        q = self.base.quantum.resource_counts()["quantum_params"] if hasattr(self.base, "quantum") else 0
        other = sum(p.numel() for n, p in self.base.named_parameters()
                    if p.requires_grad and not n.startswith("quantum."))
        return {"generated_weights": self.n_generated, "qt_qubits": self.generator.n_qubits,
                "generator_params": gen, "affine_params": aff,
                "other_classical_params": other, "quantum_params": q,
                "total_trainable": gen + aff + other + q}

    @torch.no_grad()
    def materialize(self) -> nn.Module:
        """Copy generated weights into the base model and return it (QNN no longer needed)."""
        params = self.generate()
        for name, w in params.items():
            p = self.base.get_parameter(name)
            p.copy_(w)
            p.requires_grad_(True)
        return self.base
