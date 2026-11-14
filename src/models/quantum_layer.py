"""
Variational quantum circuit (VQC) layer for the HQ-GNN.

Compared with the mid-term version this layer is
  * batched      - one QNode call per mini-batch via PennyLane parameter
                   broadcasting, instead of a Python loop over crystals;
  * prunable     - every rotation angle has a boolean mask entry. Masked
                   angles are not just zeroed, their gates are *removed* from
                   the circuit, so gate count and depth really shrink;
  * honest about dead parameters - with a CNOT ring, the final RZ of every
                   qubit in the last layer has exactly zero effect on <Z>
                   (diagonal gate followed only by a basis permutation and a
                   Z measurement). Those 8 angles are excluded by default;
  * optionally entangler-prunable - `entangler="crx"` replaces each CNOT with
                   a trainable CRX(gamma). CRX(0) = identity, so two-qubit
                   gates can be pruned by the same machinery as rotations.

The circuit is mathematically identical to the old one when
entangler="cnot": qml.Rot(phi, theta, omega) == RZ(omega) RY(theta) RZ(phi),
which is exactly the RZ -> RY -> RZ sequence applied below. Old checkpoints
(`weights` of shape [L, N, 3]) therefore load unchanged (strict=False because
of the new mask buffers).
"""
import math
from typing import Dict, Optional

import pennylane as qml
import torch
import torch.nn as nn


class QuantumCircuitLayer(nn.Module):
    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 3,
        diff_method: str = "backprop",
        entangler: str = "cnot",          # "cnot" (fixed) or "crx" (trainable, prunable)
        input_scaling: str = "none",      # "none" or "tanh" (angles = pi * tanh(x))
        drop_dead_params: bool = True,
        batched: bool = True,
    ):
        super().__init__()
        if entangler not in ("cnot", "crx"):
            raise ValueError(f"entangler must be 'cnot' or 'crx', got {entangler!r}")
        if input_scaling not in ("none", "tanh"):
            raise ValueError(f"input_scaling must be 'none' or 'tanh', got {input_scaling!r}")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entangler = entangler
        self.input_scaling = input_scaling
        self.drop_dead_params = drop_dead_params
        self.batched = batched

        # (layer, qubit, [phi, theta, omega]) -> gates RZ(phi), RY(theta), RZ(omega)
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.01)
        self.register_buffer("mask", torch.ones(n_layers, n_qubits, 3, dtype=torch.bool))

        if entangler == "crx":
            # CRX(pi) is CNOT up to a local phase, so start close to the old circuit.
            self.ent_weights = nn.Parameter(math.pi + 0.01 * torch.randn(n_layers, n_qubits))
        else:
            self.ent_weights = None
        self.register_buffer("ent_mask", torch.ones(n_layers, n_qubits, dtype=torch.bool))

        # Python-side copies of the masks, read while the QNode builds its tape.
        self._active = None
        self._ent_active = None

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def _circuit(inputs, weights, ent_weights=None):
            n, L = self.n_qubits, self.n_layers
            act, ent_act = self._active, self._ent_active

            # Angle encoding. `inputs[..., i]` is shape (B,) when batched, so
            # PennyLane broadcasts the whole circuit over the batch.
            for i in range(n):
                qml.RY(inputs[..., i], wires=i)

            for l in range(L):
                for i in range(n):
                    if act[l][i][0]:
                        qml.RZ(weights[l, i, 0], wires=i)
                    if act[l][i][1]:
                        qml.RY(weights[l, i, 1], wires=i)
                    if act[l][i][2]:
                        qml.RZ(weights[l, i, 2], wires=i)
                if n > 1:
                    for i in range(n):
                        j = (i + 1) % n
                        if self.entangler == "cnot":
                            qml.CNOT(wires=[i, j])
                        elif ent_act[l][i]:
                            qml.CRX(ent_weights[l, i], wires=[i, j])

            return [qml.expval(qml.PauliZ(i)) for i in range(n)]

        self.circuit = _circuit

    # ------------------------------------------------------------------ masks
    def dead_mask(self) -> torch.Tensor:
        """Angles that provably cannot change any <Z_i> output.

        The last-layer omega (final RZ) of qubit i is dead when every gate
        after it on wire i is either diagonal on i or a computational-basis
        permutation: true for a CNOT ring, and for a CRX ring when the CRX
        that *targets* qubit i (gate index i-1) has been pruned.
        """
        dead = torch.zeros_like(self.mask)
        if not self.drop_dead_params:
            return dead
        n = self.n_qubits
        for i in range(n):
            if self.entangler == "cnot" or n == 1:
                dead[-1, i, 2] = True
            elif not bool(self.ent_mask[-1, (i - 1) % n]):
                dead[-1, i, 2] = True
        return dead

    def active_mask(self) -> torch.Tensor:
        return self.mask & ~self.dead_mask()

    def set_mask(self, mask: torch.Tensor, ent_mask: Optional[torch.Tensor] = None) -> None:
        """Install new pruning masks and zero the removed angles."""
        with torch.no_grad():
            self.mask.copy_(mask.to(self.mask.device, dtype=torch.bool))
            self.weights[~self.mask] = 0.0
            if ent_mask is not None:
                if self.entangler != "crx":
                    raise ValueError("ent_mask only applies to entangler='crx'")
                self.ent_mask.copy_(ent_mask.to(self.ent_mask.device, dtype=torch.bool))
                self.ent_weights[~self.ent_mask] = 0.0

    def _sync_active(self) -> None:
        self._active = self.active_mask().cpu().tolist()
        self._ent_active = self.ent_mask.cpu().tolist()

    # -------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_qubits) -> (batch, n_qubits) of <Z_i> in [-1, 1]."""
        out_device, out_dtype = x.device, x.dtype
        x = x.cpu()  # default.qubit simulates on CPU
        if self.input_scaling == "tanh":
            x = math.pi * torch.tanh(x)
        self._sync_active()

        extra = () if self.ent_weights is None else (self.ent_weights,)

        if self.batched:
            res = self.circuit(x, self.weights, *extra)
            out = torch.stack(list(res), dim=-1)
            if out.dim() == 1:  # some PennyLane versions squeeze a batch of 1
                out = out.unsqueeze(0)
        else:
            out = torch.stack(
                [torch.stack(list(self.circuit(x[b], self.weights, *extra))) for b in range(x.shape[0])]
            )
        return out.to(dtype=out_dtype, device=out_device)

    # ----------------------------------------------------- cost reporting
    def resource_counts(self) -> Dict[str, int]:
        """Counts for the circuit as it is currently masked.

        Gates are counted after decomposing Rot into RZ, RY, RZ (closer to
        what hardware runs). For reference, the mid-term circuit is
        72 angles, 8 + 72 = 80 one-qubit gates, 24 CNOTs, 104 gates total.
        """
        n, L = self.n_qubits, self.n_layers
        act = self.active_mask()
        n_rot = int(act.sum())
        if n > 1:
            n_2q = L * n if self.entangler == "cnot" else int(self.ent_mask.sum())
        else:
            n_2q = 0

        # Circuit depth: longest chain of gates on any wire.
        depth = [1] * n  # encoding layer
        for l in range(L):
            for i in range(n):
                depth[i] += int(act[l, i].sum())
            if n > 1:
                for i in range(n):
                    j = (i + 1) % n
                    if self.entangler == "cnot" or bool(self.ent_mask[l, i]):
                        d = max(depth[i], depth[j]) + 1
                        depth[i] = depth[j] = d

        trainable = n_rot + (int(self.ent_mask.sum()) if self.entangler == "crx" else 0)
        return {
            "quantum_params": trainable,
            "one_qubit_gates": n + n_rot,
            "two_qubit_gates": n_2q,
            "total_gates": n + n_rot + n_2q,
            "depth": max(depth),
        }
