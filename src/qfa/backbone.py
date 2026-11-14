"""
Frozen foundation-model features for QFA-Net.

A pretrained universal interatomic potential (ORB, Orbital Materials, Apache-2.0) is run
ONCE per crystal with no gradients. We keep the atom vectors after the encoder and after
every message-passing layer:   H[l] in R^{N x 256},  l = 0 .. L  (orb-v2: L = 15).
They are cached to disk; training never touches the backbone again.

Why all layers: Kim, Park & Li (HackNIP, 2025) found that the best single ORB layer
differs by task (exfoliation energy: layer 11; phonons: 3; band gap: 7). QFA-Net learns
a per-task mix of all layers instead of searching for one (see model.DepthMixer).

Storage tiers:
  atoms  : every atom, every layer, float16          (tiny/small/medium tasks)
  pooled : mean and max over atoms, every layer      (large tasks, keeps disk use < 2 GB)

Install (Python 3.10):  pip install "orb-models==0.5.5" ase   (orb-models >= 0.6 needs Python 3.12)
"""
import os
from typing import Dict, List, Sequence

import numpy as np


def load_orb(name: str = "orb_v2", device: str = "cpu"):
    """Loads a pretrained ORB model. Works with orb-models 0.5.x (Python 3.10/3.11: returns the model;
    graphs via atomic_system + model.system_config) and >= 0.6 (returns (model, atoms_adapter)).
    compile=False: torch.compile is slow or unsupported on Windows CPUs."""
    from orb_models.forcefield import pretrained  # imported lazily
    fn = getattr(pretrained, name)
    try:
        out = fn(device=device, precision="float32-high", compile=False)
    except TypeError:
        out = fn(device=device)
    if isinstance(out, tuple):
        model, adapter = out
    else:
        model, adapter = out, None
    model.eval()
    return model, adapter


def _to_graph(model, adapter, atoms, device):
    if adapter is not None:
        return adapter.from_ase_atoms(atoms, device=device)
    from orb_models.forcefield import atomic_system  # orb-models 0.5.x
    return atomic_system.ase_atoms_to_atom_graphs(atoms, model.system_config, device=device)


def _nodes(out):
    """Node features from a layer output: (nodes, edges) tuples in recent orb-models,
    AtomGraphs objects (node_features["feat"]) in older releases."""
    if isinstance(out, (tuple, list)):
        return out[0]
    nf = getattr(out, "node_features", None)
    if isinstance(nf, dict):
        return nf.get("feat", next(iter(nf.values())))
    return out


def layer_embeddings(structures: Sequence, name: str = "orb_v2", device: str = "cpu",
                     mode: str = "atoms", log_every: int = 200) -> List[np.ndarray]:
    """Returns one array per crystal: (L+1, N, 256) float16 for mode='atoms',
    or (L+1, 512) float32 [mean | max] for mode='pooled'."""
    import torch
    from pymatgen.io.ase import AseAtomsAdaptor

    model, adapter = load_orb(name, device)
    gns = model.model
    store: List = []
    hooks = [gns._encoder.register_forward_hook(lambda m, i, o: store.append(_nodes(o).detach()))]
    for layer in gns.gnn_stacks:
        hooks.append(layer.register_forward_hook(lambda m, i, o: store.append(_nodes(o).detach())))
    out = []
    try:
        for k, s in enumerate(structures):
            atoms = AseAtomsAdaptor.get_atoms(s)
            graph = _to_graph(model, adapter, atoms, device)
            store.clear()
            with torch.no_grad():
                gns(graph)
            H = torch.stack(store, dim=0).float().cpu()          # (L+1, N, 256)
            if mode == "atoms":
                out.append(H.numpy().astype(np.float16))
            else:
                out.append(torch.cat([H.mean(1), H.max(1).values], dim=1).numpy())
            if log_every and (k + 1) % log_every == 0:
                print(f"  backbone: {k + 1}/{len(structures)} crystals", flush=True)
    finally:
        for h in hooks:
            h.remove()
    return out


def cached_embeddings(task: str, structures: Sequence, ids: Sequence[str], cache_dir: str,
                      name: str = "orb_v2", mode: str = "atoms") -> Dict[str, np.ndarray]:
    """Computes once, then loads from data/cache/qfa_<task>_<name>_<mode>_v1.npz."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"qfa_{task}_{name}_{mode}_v1.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return {i: z[i] for i in z.files}
    import json
    import time
    t0 = time.time()
    embs = layer_embeddings(structures, name=name, mode=mode)
    data = {i: e for i, e in zip(ids, embs)}
    np.savez(path, **data)
    with open(path.replace(".npz", "_time.json"), "w") as fh:   # backbone cost, reported in the results
        json.dump({"seconds": time.time() - t0, "crystals": len(ids)}, fh)
    return data
