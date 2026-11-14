"""Tests for the batched/prunable VQC, pruning loop, Quantum-Train and size policy."""
import copy
import math

import pennylane as qml
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


# ----------------------------------------------------------------- helpers
def _synthetic(n=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        k = int(torch.randint(2, 6, (1,), generator=g))
        x = torch.zeros(k, 92)
        x[torch.arange(k), torch.randint(0, 92, (k,), generator=g)] = 1.0
        src = list(range(k - 1)); dst = list(range(1, k))
        ei = torch.tensor([src + dst, dst + src], dtype=torch.long)
        ea = torch.rand(len(src) * 2, 41, generator=g)
        out.append(Data(x=x, edge_index=ei, edge_attr=ea, y=torch.tensor([(k - 3.5) / 1.2])))
    return out


def _cfg():
    return {
        "gnn": {"node_input_dim": 92, "edge_input_dim": 41, "hidden_dim": 16, "num_layers": 1},
        "quantum": {"n_qubits": 4, "n_layers": 2, "diff_method": "backprop"},
        "mlp": {"hidden_dim": 8, "dropout": 0.0, "skip_connection": True},
        "training": {"lr": 0.01, "epochs": 3, "patience": 3, "seed": 0},
        "pruning": {"method": "ablation", "fraction_per_round": 0.25, "max_sparsity": 0.5,
                    "tolerance": 10.0, "finetune_epochs": 2},
        "qt": {"targets": ["gnn", "mlp"], "n_qnn_layers": 2, "mapping_hidden": 8},
    }


# ------------------------------------------------------------ VQC layer
class TestPrunableVQC:
    def test_matches_old_rot_circuit(self):
        """Decomposed RZ-RY-RZ circuit == the mid-term qml.Rot circuit."""
        from src.models.quantum_layer import QuantumCircuitLayer
        n, L = 4, 2
        layer = QuantumCircuitLayer(n, L, drop_dead_params=False)
        with torch.no_grad():
            layer.weights.uniform_(-math.pi, math.pi)
        dev = qml.device("default.qubit", wires=n)

        @qml.qnode(dev, interface="torch")
        def old(inputs, w):
            for i in range(n):
                qml.RY(inputs[i], wires=i)
            for l in range(L):
                for i in range(n):
                    qml.Rot(w[l, i, 0], w[l, i, 1], w[l, i, 2], wires=i)
                for i in range(n):
                    qml.CNOT(wires=[i, (i + 1) % n])
            return [qml.expval(qml.PauliZ(i)) for i in range(n)]

        x = torch.randn(3, n)
        new_out = layer(x)
        old_out = torch.stack([torch.stack(list(old(x[b], layer.weights))) for b in range(3)]).float()
        torch.testing.assert_close(new_out, old_out, atol=1e-5, rtol=1e-5)

    def test_batched_equals_loop(self):
        from src.models.quantum_layer import QuantumCircuitLayer
        a = QuantumCircuitLayer(4, 2, batched=True)
        b = QuantumCircuitLayer(4, 2, batched=False)
        b.load_state_dict(a.state_dict())
        x = torch.randn(5, 4)
        torch.testing.assert_close(a(x), b(x), atol=1e-5, rtol=1e-5)

    def test_last_layer_omega_is_dead(self):
        """With a CNOT ring the final RZ of every qubit has zero gradient."""
        from src.models.quantum_layer import QuantumCircuitLayer
        layer = QuantumCircuitLayer(4, 2, drop_dead_params=False)
        with torch.no_grad():
            layer.weights.uniform_(-math.pi, math.pi)
        layer(torch.randn(6, 4)).sum().backward()
        g = layer.weights.grad
        assert torch.allclose(g[-1, :, 2], torch.zeros(4), atol=1e-7)
        assert g[:-1].abs().sum() > 0

    def test_dropping_dead_params_changes_nothing(self):
        from src.models.quantum_layer import QuantumCircuitLayer
        full = QuantumCircuitLayer(4, 2, drop_dead_params=False)
        lean = QuantumCircuitLayer(4, 2, drop_dead_params=True)
        with torch.no_grad():
            full.weights.uniform_(-math.pi, math.pi)
        lean.load_state_dict(full.state_dict())
        x = torch.randn(4, 4)
        torch.testing.assert_close(full(x), lean(x), atol=1e-5, rtol=1e-5)
        assert lean.resource_counts()["quantum_params"] == 2 * 4 * 3 - 4

    def test_mask_removes_gates(self):
        from src.models.quantum_layer import QuantumCircuitLayer
        layer = QuantumCircuitLayer(4, 2)
        before = layer.resource_counts()
        m = layer.mask.clone()
        m[0, :, 0] = False
        layer.set_mask(m)
        after = layer.resource_counts()
        assert after["quantum_params"] == before["quantum_params"] - 4
        assert after["one_qubit_gates"] == before["one_qubit_gates"] - 4
        assert torch.all(layer.weights[0, :, 0] == 0)

    def test_crx_entanglers_prunable(self):
        from src.models.quantum_layer import QuantumCircuitLayer
        layer = QuantumCircuitLayer(4, 2, entangler="crx")
        assert layer.resource_counts()["two_qubit_gates"] == 8
        em = layer.ent_mask.clone(); em[1, 0] = False
        layer.set_mask(layer.mask, em)
        assert layer.resource_counts()["two_qubit_gates"] == 7
        out = layer(torch.randn(2, 4))
        assert out.shape == (2, 4)


# --------------------------------------------------------------- pruning
class TestPruning:
    def test_prune_lowest_counts(self):
        from src.models.quantum_layer import QuantumCircuitLayer
        from src.optimization.pruning import prune_lowest
        layer = QuantumCircuitLayer(4, 2)
        n0 = layer.resource_counts()["quantum_params"]
        scores = {"rot": torch.rand(layer.weights.shape), "ent": None}
        scores["rot"][~layer.active_mask()] = float("inf")
        prune_lowest(layer, scores, 5)
        assert layer.resource_counts()["quantum_params"] == n0 - 5

    @pytest.mark.parametrize("method", ["ablation", "magnitude", "taylor"])
    def test_scores_shape(self, method):
        from src.models.hybrid_model import HybridQuantumGNN
        from src.optimization.pruning import collect_latents, score_angles
        model = HybridQuantumGNN(_cfg())
        X, y = collect_latents(model, DataLoader(_synthetic(8), batch_size=4))
        s = score_angles(model, X, y, method)
        act = model.quantum.active_mask()
        assert torch.isfinite(s["rot"][act]).all()
        assert torch.isinf(s["rot"][~act]).all()

    def test_iterative_prune_runs(self):
        from src.models.hybrid_model import HybridQuantumGNN
        from src.optimization.pruning import iterative_prune
        cfg = _cfg()
        model = HybridQuantumGNN(cfg)
        data = _synthetic(24)
        tr, va = DataLoader(data[:16], batch_size=8, shuffle=True), DataLoader(data[16:], batch_size=8)
        n0 = model.quantum.resource_counts()["quantum_params"]
        hist = iterative_prune(model, cfg, tr, va, 0.0, 1.0)
        n1 = model.quantum.resource_counts()["quantum_params"]
        assert len(hist) >= 2
        assert math.ceil(0.5 * n0) <= n1 < n0


# ---------------------------------------------------------- Quantum-Train
class TestQuantumTrain:
    def test_generator_size(self):
        from src.optimization.quantum_train import QTGenerator
        g = QTGenerator(1000, n_qnn_layers=2, mapping_hidden=8)
        assert g.n_qubits == 10
        assert g().shape == (1000,)

    @pytest.mark.parametrize("gen", ["quantum", "classical", "none"])
    @pytest.mark.parametrize("mode", ["full", "delta"])
    def test_wrapper_trains_only_generator(self, gen, mode):
        from src.models.hybrid_model import HybridQuantumGNN
        from src.optimization.quantum_train import QuantumTrainModel
        cfg = _cfg()
        base = HybridQuantumGNN(cfg)
        qt = QuantumTrainModel(base, {**cfg["qt"], "generator": gen, "mode": mode})
        batch = next(iter(DataLoader(_synthetic(6), batch_size=6)))
        out = qt(batch)
        assert out.shape == (6,)
        ((out - batch.y.view(-1)) ** 2).mean().backward()
        assert base.gnn.embedding.weight.grad is None            # generated, not trained
        assert qt.generator.mapping[0].weight.grad is not None   # trained
        assert base.quantum.weights.grad is not None             # circuit still trained
        bd = qt.trainable_breakdown()
        assert bd["total_trainable"] < sum(p.numel() for p in base.parameters())

    def test_materialize_matches(self):
        from src.models.hybrid_model import HybridQuantumGNN
        from src.optimization.quantum_train import QuantumTrainModel
        cfg = _cfg()
        qt = QuantumTrainModel(HybridQuantumGNN(cfg), cfg["qt"])
        batch = next(iter(DataLoader(_synthetic(6), batch_size=6)))
        qt.eval()
        with torch.no_grad():
            a = qt(batch)
        base = qt.materialize()
        base.eval()
        with torch.no_grad():
            b = base(batch)
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------ size policy
def test_size_policy_tiers():
    from src.optimization.size_policy import resolve_profile
    assert resolve_profile(509, {})["adaptive"]["resolved_tier"] == "tiny"
    assert resolve_profile(4000, {})["qt"]["enabled"] is True
    assert resolve_profile(17000, {})["qt"]["enabled"] is False
    big = resolve_profile(106000, {"quantum": {"n_layers": 3}})
    assert big["adaptive"]["resolved_tier"] == "large" and big["quantum"]["n_layers"] == 4
