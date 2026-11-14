"""
QFA-Net: Quantum-compressed Foundation-Adapter Network.

Part 1, classical (the new part):
  (a) frozen foundation features  H[l] (N x 256) for l = 0..L, cached once (backbone.py)
  (b) DepthMixer: K learned softmax mixes over the L+1 layers (per task)      -> N x 256K
  (c) down-projection 256K -> d                                               -> N x d
  (d) GraphAdapter: 2 gated, residual message-passing layers on our own
      image-aware 12-neighbour graph (8 A), only these are trained            -> N x d
  (e) pooling: attention-weighted sum | mean | max -> Linear 3d -> d          -> g (d)
  (f) descriptor branch: Magpie 132 + symmetry 8 -> MI top-k -> MLP           -> c (32)
  (g) h = [g | c];  classical head MLP(h) -> y_c
Part 2, quantum:
  u = Linear(h) (8), x = pi * tanh(u), 8-qubit re-uploading circuit (3 layers),
  16 readouts -> Linear (zero init) -> r ;   y = y_c + r   (r = 0 at start)
  + gate pruning (train.py) and Quantum-Train (QTGeneratorTorch) for the trainable weights.
Large tier ("pooled" cache): (b)-(e) are replaced by DepthMixer on pooled [mean|max] vectors.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantum_torch import QuantumHeadTorch


class DepthMixer(nn.Module):
    """K softmax-weighted averages over layers of layer-normalised features."""

    def __init__(self, n_layers: int, dim: int, k: int = 2, last_only: bool = False):
        super().__init__()
        self.last_only = last_only
        self.alpha = nn.Parameter(torch.zeros(k, n_layers))
        with torch.no_grad():  # start mix 1 on the deep half, mix 2 on the shallow half
            half = n_layers // 2
            self.alpha[0, half:] = 1.0
            if k > 1:
                self.alpha[1, :half] = 1.0
        self.dim = dim

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.alpha, dim=-1)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """H: (L+1, N, D) -> (N, K*D); ablation last_only: (N, D) from the deepest layer"""
        if self.last_only:
            return F.layer_norm(H[-1].float(), (H.shape[-1],))
        Hn = F.layer_norm(H.float(), (H.shape[-1],))
        w = self.weights()                                   # (K, L+1)
        mixed = torch.einsum("kl,lnd->knd", w, Hn)           # (K, N, D)
        return mixed.permute(1, 0, 2).reshape(H.shape[1], -1)


class GatedAdapterLayer(nn.Module):
    """h_i <- LN(h_i + sum_j sigmoid(W_g [h_i|h_j|e_ij]) * (W_v h_j)) / sqrt(deg)"""

    def __init__(self, d: int, d_edge: int):
        super().__init__()
        self.gate = nn.Linear(2 * d + d_edge, d)
        self.value = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, h, src, dst, e):
        z = torch.cat([h[dst], h[src], e], dim=-1)
        msg = torch.sigmoid(self.gate(z)) * self.value(h[src])
        agg = torch.zeros_like(h).index_add_(0, dst, msg)
        deg = torch.zeros(h.shape[0], device=h.device).index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
        return self.norm(h + agg / deg.clamp(min=1).sqrt()[:, None])


def rbf(d: torch.Tensor, n: int = 16, cutoff: float = 8.0) -> torch.Tensor:
    mu = torch.linspace(0, cutoff, n, device=d.device)
    gamma = (n / cutoff) ** 2
    return torch.exp(-gamma * (d[:, None] - mu[None, :]) ** 2)


class QFANet(nn.Module):
    def __init__(self, n_layers: int, emb_dim: int = 256, d: int = 64, n_desc: int = 64,
                 k_mix: int = 2, n_adapter: int = 2, mode: str = "atoms", use_graph: bool = True,
                 n_qubits: int = 8, q_layers: int = 3, quantum: bool = True, q_kind: str = "circuit",
                 ablation: str = "none"):
        super().__init__()
        self.mode, self.use_graph, self.quantum, self.q_kind = mode, use_graph, quantum, q_kind
        hidden_c = 32
        self.ablation = ablation
        self.use_desc = ablation != "no_desc" or not use_graph
        if ablation == "no_adapter":
            n_adapter = 0
        if use_graph:
            last = ablation == "last_layer"
            self.mixer = DepthMixer(n_layers, emb_dim if mode == "atoms" else 2 * emb_dim, k_mix, last_only=last)
            in_dim = (1 if last else k_mix) * (emb_dim if mode == "atoms" else 2 * emb_dim)
            self.down = nn.Linear(in_dim, d)
            if mode == "atoms":
                self.adapter = nn.ModuleList([GatedAdapterLayer(d, 16) for _ in range(n_adapter)])
                self.att = nn.Linear(d, 1)
                self.readout = nn.Linear(3 * d, d)
            g_dim = d
        else:
            g_dim = 0
        self.desc = nn.Sequential(nn.Linear(n_desc, hidden_c), nn.SiLU(), nn.Linear(hidden_c, hidden_c), nn.SiLU())
        self.h_dim = g_dim + (hidden_c if self.use_desc else 0)
        self.head = nn.Sequential(nn.Linear(self.h_dim, 64), nn.SiLU(), nn.Linear(64, 1))
        # quantum residual
        self.q_in = nn.Linear(self.h_dim, n_qubits)
        self.circuit = QuantumHeadTorch(n_qubits, q_layers)
        self.circuit.drop_dead()
        if q_kind == "mlp":   # classical control of about the same size as the circuit
            self.ctrl = nn.Sequential(nn.Linear(n_qubits, 2 * n_qubits), nn.SiLU())
        self.q_out = nn.Linear(self.circuit.n_outputs, 1)
        nn.init.zeros_(self.q_out.weight)
        nn.init.zeros_(self.q_out.bias)

    # --------------------------------------------------------------------- encoders
    def graph_vector(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.mode == "pooled":
            return F.silu(self.down(self.mixer(batch["H"]).reshape(batch["H"].shape[1], -1)))
        x = F.silu(self.down(self.mixer(batch["H"])))              # (N_total, d)
        e = rbf(batch["dist"])
        for layer in self.adapter:
            x = layer(x, batch["src"], batch["dst"], e)
        b, n_graphs = batch["atom_graph"], batch["n_graphs"]
        a = self.att(x)[:, 0]
        a = torch.exp(a - a.max())
        den = torch.zeros(n_graphs, device=x.device).index_add_(0, b, a)
        att = torch.zeros(n_graphs, x.shape[1], device=x.device).index_add_(0, b, a[:, None] * x)
        att = att / den[:, None].clamp(min=1e-12)
        cnt = torch.zeros(n_graphs, device=x.device).index_add_(0, b, torch.ones_like(a))
        mean = torch.zeros(n_graphs, x.shape[1], device=x.device).index_add_(0, b, x) / cnt[:, None]
        mx = torch.full((n_graphs, x.shape[1]), -1e9, device=x.device).scatter_reduce(
            0, b[:, None].expand_as(x), x, reduce="amax", include_self=True)
        return F.silu(self.readout(torch.cat([att, mean, mx], dim=1)))

    def latent(self, batch) -> torch.Tensor:
        if not self.use_graph:
            return self.desc(batch["desc"])
        g = self.graph_vector(batch)
        return torch.cat([g, self.desc(batch["desc"])], dim=1) if self.use_desc else g

    # --------------------------------------------------------------------- outputs
    def forward(self, batch, return_parts: bool = False):
        h = self.latent(batch)
        y_c = self.head(h)[:, 0]
        if not self.quantum:
            return (y_c, y_c, torch.zeros_like(y_c)) if return_parts else y_c
        x = torch.pi * torch.tanh(self.q_in(h))
        feats = self.ctrl(x / torch.pi) if self.q_kind == "mlp" else self.circuit(x)
        r = self.q_out(feats)[:, 0]
        return (y_c + r, y_c, r) if return_parts else y_c + r

    def classical_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith(("q_in", "circuit", "q_out", "ctrl"))]

    def quantum_parameters(self):
        return [p for n, p in self.named_parameters() if n.startswith(("q_in", "circuit", "q_out", "ctrl"))]

    def n_trainable(self) -> Dict[str, int]:
        cl = sum(p.numel() for p in self.classical_parameters())
        q = sum(p.numel() for n, p in self.named_parameters() if n.startswith(("q_in", "q_out", "ctrl")))
        circ = self.circuit.n_active() if self.q_kind == "circuit" else 0
        return {"classical": cl, "quantum_head": q + circ, "total": cl + q + circ}
