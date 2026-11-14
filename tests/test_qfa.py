"""PyTorch checks of the full QFA-Net (run on a machine with torch installed)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.qfa.lab.qsim import QuantumHead  # noqa: E402
from src.qfa.model import QFANet  # noqa: E402
from src.qfa.quantum_torch import QTGeneratorTorch, QuantumHeadTorch  # noqa: E402


def test_torch_circuit_matches_numpy():
    rng = np.random.default_rng(0)
    ref = QuantumHead(5, 3, rng, init_scale=1.0)
    ref.drop_dead()
    ref.mask[1, 2, 0] = False
    ref.a[1, 2, 0] = 0.0
    th = QuantumHeadTorch(5, 3).double()
    with torch.no_grad():
        th.a.copy_(torch.tensor(ref.a))
        th.mask.copy_(torch.tensor(ref.mask))
    x = rng.normal(size=(4, 5))
    out_np, _ = ref.forward(x)
    out_t = th(torch.tensor(x)).detach().numpy()
    assert np.abs(out_np - out_t).max() < 1e-10


def test_qt_generator_size_and_grad():
    g = QTGeneratorTorch([(8, 10), (8,), (1, 8), (1,)], n_layers=3)
    w = g.generate()
    assert w.shape == (97,) and g.n == 7
    w.sum().backward()
    assert g.phi.grad is not None and torch.isfinite(g.phi.grad).all()


def _fake_batch(n_graphs=3, layers=6, emb=256, n_desc=20):
    sizes = [4, 6, 3][:n_graphs]
    N = sum(sizes)
    src, dst, gid, off = [], [], [], 0
    for k, n in enumerate(sizes):
        for i in range(n):
            for j in range(n):
                if i != j:
                    src.append(off + j)
                    dst.append(off + i)
        gid += [k] * n
        off += n
    return {"H": torch.randn(layers, N, emb), "src": torch.tensor(src), "dst": torch.tensor(dst),
            "dist": 2 + 3 * torch.rand(len(src)), "atom_graph": torch.tensor(gid), "n_graphs": n_graphs,
            "desc": torch.randn(n_graphs, n_desc)}


def test_model_forward_and_twin():
    b = _fake_batch()
    m = QFANet(n_layers=6, n_desc=20)
    y, yc, r = m(b, return_parts=True)
    assert y.shape == (3,)
    assert torch.allclose(r, torch.zeros_like(r))       # quantum residual starts at 0: classical twin
    loss = y.sum()
    loss.backward()
    assert m.mixer.alpha.grad is not None


def test_pooled_and_descriptor_only_modes():
    m = QFANet(n_layers=6, n_desc=20, mode="pooled")
    b = {"H": torch.randn(6, 5, 512), "desc": torch.randn(5, 20)}
    assert m(b).shape == (5,)
    m2 = QFANet(n_layers=1, n_desc=20, use_graph=False)
    assert m2({"desc": torch.randn(5, 20)}).shape == (5,)


@pytest.mark.parametrize("ablation", ["last_layer", "no_adapter", "no_desc"])
def test_ablations_forward(ablation):
    b = _fake_batch()
    m = QFANet(n_layers=6, n_desc=20, ablation=ablation)
    assert m(b).shape == (3,)
