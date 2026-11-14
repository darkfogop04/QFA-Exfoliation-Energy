import math

import torch
import torch.nn as nn
from src.models.gnn import GNNBackbone
from src.models.quantum_layer import QuantumCircuitLayer


class HybridQuantumGNN(nn.Module):
    """
    Hybrid Quantum-Classical Graph Neural Network.
    Combines a CGCNN backbone, a PennyLane VQC, and an MLP head.

    The forward pass is split into `encode` (graph -> latent x) and
    `head` (latent x -> prediction). Pruning uses this split: the GNN does not
    depend on the circuit angles, so latents can be computed once and reused
    while the circuit is ablated angle by angle.
    """
    def __init__(self, config: dict):
        super().__init__()
        gnn_cfg = config.get('gnn', {})
        q_cfg = config.get('quantum', {})
        mlp_cfg = config.get('mlp', {})

        n_qubits = q_cfg.get('n_qubits', 8)
        self.skip_connection = mlp_cfg.get('skip_connection', False)
        self.l1_angles = q_cfg.get('l1_angles', 0.0)

        # 1. GNN Backbone
        self.gnn = GNNBackbone(
            node_input_dim=gnn_cfg.get('node_input_dim', 92),
            edge_input_dim=gnn_cfg.get('edge_input_dim', 41),
            hidden_dim=gnn_cfg.get('hidden_dim', 64),
            num_layers=gnn_cfg.get('num_layers', 3),
            output_dim=n_qubits
        )

        # 2. Quantum Layer
        self.quantum = QuantumCircuitLayer(
            n_qubits=n_qubits,
            n_layers=q_cfg.get('n_layers', 3),
            diff_method=q_cfg.get('diff_method', 'backprop'),
            entangler=q_cfg.get('entangler', 'cnot'),
            input_scaling=q_cfg.get('input_scaling', 'none'),
            drop_dead_params=q_cfg.get('drop_dead_params', True),
            batched=q_cfg.get('batched', True),
        )

        # 3. MLP Head
        input_dim = n_qubits * 2 if self.skip_connection else n_qubits
        mlp_hidden = mlp_cfg.get('hidden_dim', 32)
        dropout = mlp_cfg.get('dropout', 0.1)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
        )

    def regularization_loss(self) -> torch.Tensor:
        """lambda * sum |wrapped active angle|; 0 unless quantum.l1_angles > 0."""
        if self.l1_angles <= 0:
            return torch.zeros((), device=self.quantum.weights.device)
        w = torch.remainder(self.quantum.weights + math.pi, 2 * math.pi) - math.pi
        return self.l1_angles * w[self.quantum.active_mask()].abs().sum()

    def encode(self, data) -> torch.Tensor:
        """Graph batch -> latent vector, shape (batch_size, n_qubits)."""
        return self.gnn(data.x, data.edge_index, data.edge_attr, data.batch)

    def head(self, gnn_out: torch.Tensor) -> torch.Tensor:
        """Latent vector -> prediction, shape (batch_size,)."""
        q_out = self.quantum(gnn_out)
        combined = torch.cat([gnn_out, q_out], dim=-1) if self.skip_connection else q_out
        combined = combined.to(next(self.mlp.parameters()).device)
        return self.mlp(combined).squeeze(-1)

    def forward(self, data) -> torch.Tensor:
        return self.head(self.encode(data))
