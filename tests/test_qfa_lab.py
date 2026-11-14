"""NumPy-only checks of the QFA-Net lab code (no PyTorch needed)."""
import numpy as np
import pytest

from src.qfa.lab.featurize import magpie_features, parse_formula
from src.qfa.lab.nn import MLP
from src.qfa.lab.qsim import QuantumHead
from src.qfa.lab.qt import QTGenerator


def test_parse_formula():
    assert parse_formula("Ag(AuS)2") == {"Ag": 1.0, "Au": 2.0, "S": 2.0}
    assert parse_formula("Hf2 Si2 Te2") == {"Hf": 2.0, "Si": 2.0, "Te": 2.0}
    assert parse_formula("Fe0.62C0.001")["C"] == pytest.approx(0.001)


def test_magpie_mos2_by_hand():
    x = magpie_features("Mo1 S2")
    # Number: min 16, max 42, range 26, mean 74/3, mode = S (largest fraction) = 16
    assert x[:4] == pytest.approx([16, 42, 26, 74 / 3])
    assert x[5] == 16
    assert x.shape == (132,)


def test_one_qubit_example():
    h = QuantumHead(2, 1)
    h.a[:] = 0.0
    out, _ = h.forward(np.array([[0.8, 0.0]]))
    assert out[0, 0] == pytest.approx(np.cos(0.8))       # <Z> = 0.697


def test_circuit_adjoint_gradients():
    rng = np.random.default_rng(1)
    h = QuantumHead(4, 3, rng, init_scale=1.0)
    h.drop_dead()
    x = rng.normal(size=(3, 4))
    g = rng.normal(size=(3, 8))
    _, psi = h.forward(x)
    da, dx = h.backward(x, psi, g)
    eps = 1e-6
    for idx in [(0, 1, 0), (1, 2, 1), (2, 3, 0)]:
        h.a[idx] += eps
        lp = (h.forward(x)[0] * g).sum()
        h.a[idx] -= 2 * eps
        lm = (h.forward(x)[0] * g).sum()
        h.a[idx] += eps
        assert da[idx] == pytest.approx((lp - lm) / (2 * eps), abs=1e-6)


def test_dead_gates_have_no_effect():
    rng = np.random.default_rng(2)
    h = QuantumHead(4, 2, rng, init_scale=1.0)
    x = rng.normal(size=(2, 4))
    o1, _ = h.forward(x)
    h.a[-1, :, 2] += 0.9
    o2, _ = h.forward(x)
    assert np.abs(o1 - o2).max() < 1e-12


def test_pruned_gate_is_identity():
    rng = np.random.default_rng(3)
    h = QuantumHead(4, 2, rng, init_scale=1.0)
    x = rng.normal(size=(2, 4))
    h.a[0, 1, 1] = 0.0
    o1, _ = h.forward(x)
    h.mask[0, 1, 1] = False
    o2, _ = h.forward(x)
    assert np.abs(o1 - o2).max() < 1e-12


def test_quantum_train_gradients_and_size():
    rng = np.random.default_rng(0)
    net = MLP([10, 8, 4, 1], rng)
    G = QTGenerator([p.shape for p in net.params], rng, n_layers=3)
    assert G.P == net.n_params and 2 ** G.n >= G.P > 2 ** (G.n - 1)
    gw = rng.normal(size=G.P)
    G.generate()
    g_phi, _, _ = G.backward(gw)
    eps = 1e-6
    G.phi[1, 2] += eps
    lp = G.generate() @ gw
    G.phi[1, 2] -= 2 * eps
    lm = G.generate() @ gw
    G.phi[1, 2] += eps
    assert g_phi[1, 2] == pytest.approx((lp - lm) / (2 * eps), abs=1e-6)
