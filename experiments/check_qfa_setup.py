"""Five-minute pre-flight check before the long QFA-Net runs (run on the laptop).

    python experiments/check_qfa_setup.py

1. imports, 2. ORB-v2 loads and gives 16 layer vectors for one jdft2d crystal, 3. the model runs
forward + backward on 4 real crystals, 4. the torch circuit equals the tested NumPy circuit.
Every step prints OK or the error to send back.
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def step(name, fn):
    t = time.time()
    try:
        out = fn()
        print(f"[OK]   {name} ({time.time() - t:.1f}s) {out if out is not None else ''}", flush=True)
        return out
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


def main():
    step("imports", lambda: __import__("torch").__version__ + " / orb_models " +
         getattr(__import__("orb_models"), "__version__", "?"))
    from matbench.bench import MatbenchBenchmark
    mb = MatbenchBenchmark(autoload=False, subset=["matbench_jdft2d"])
    task = list(mb.tasks)[0]
    step("load jdft2d", lambda: task.load() or len(task.df))
    structs = task.df["structure"].tolist()[:4]
    from src.qfa.backbone import layer_embeddings
    embs = step("ORB-v2 layer vectors", lambda: layer_embeddings(structs, log_every=0))
    print("       shapes:", [e.shape for e in embs])
    import numpy as np
    import torch
    from src.qfa.data import collate, descriptors, image_aware_edges
    from src.qfa.model import QFANet
    ids = [f"c{i}" for i in range(4)]
    prep = {"ids": ids, "structural": True, "mode": "atoms", "emb": dict(zip(ids, embs)),
            "graph": {i: image_aware_edges(s) for i, s in zip(ids, structs)}}
    D = step("descriptors", lambda: descriptors(structs, True))
    D = (D - D.mean(0)) / (D.std(0) + 1e-9)

    def fwd():
        m = QFANet(n_layers=embs[0].shape[0], n_desc=D.shape[1])
        b = collate(prep, np.arange(4), D.astype(np.float32))
        y = m(b)
        y.sum().backward()
        return f"output {tuple(y.shape)}, trained numbers {m.n_trainable()}"
    step("model forward + backward", fwd)

    def circ():
        from src.qfa.lab.qsim import QuantumHead
        from src.qfa.quantum_torch import QuantumHeadTorch
        rng = np.random.default_rng(0)
        ref = QuantumHead(8, 3, rng, init_scale=1.0)
        th = QuantumHeadTorch(8, 3).double()
        with torch.no_grad():
            th.a.copy_(torch.tensor(ref.a))
        x = rng.normal(size=(5, 8))
        err = np.abs(ref.forward(x)[0] - th(torch.tensor(x)).detach().numpy()).max()
        assert err < 1e-9, err
        return f"max difference {err:.1e}"
    step("torch circuit = NumPy circuit", circ)
    print("\nAll checks passed. Next: python experiments/run_qfa.py --task jdft2d --folds 0 --quick")


if __name__ == "__main__":
    main()
