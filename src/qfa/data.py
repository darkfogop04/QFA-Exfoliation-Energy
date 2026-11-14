"""
Data for the full QFA-Net (runs where PyTorch, pymatgen, matbench and orb-models are installed).

Per crystal we cache:
  * foundation features: ORB atom vectors of every layer (backbone.py)
  * our adapter graph: every atom -> its 12 nearest neighbours within 8 A, each periodic image
    kept as its own edge (no merging of images)
  * descriptors: Magpie 132 + symmetry 8 (same code as the lab, src/qfa/lab/featurize.py)
Composition-only tasks (steels, expt_gap, expt_is_metal, glass) get descriptors only.
"""
import os
from typing import Dict, List, Sequence

import numpy as np
import torch

from .lab.featurize import featurize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE = os.path.join(ROOT, "data", "cache")
COMPOSITION_ONLY = {"steels", "expt_gap", "expt_is_metal", "glass"}
CLASSIFICATION = {"expt_is_metal", "glass", "mp_is_metal"}
LARGE = {"mp_gap", "mp_is_metal", "mp_e_form"}


def image_aware_edges(structure, cutoff: float = 8.0, max_nb: int = 12):
    src, dst, dist = [], [], []
    for i, nbs in enumerate(structure.get_all_neighbors(cutoff)):
        for nb in sorted(nbs, key=lambda n: n.nn_distance)[:max_nb]:
            src.append(nb.index)
            dst.append(i)
            dist.append(nb.nn_distance)
    if not src:
        n = len(structure)
        src, dst, dist = list(range(n)), list(range(n)), [0.0] * n
    return np.array(src, np.int32), np.array(dst, np.int32), np.array(dist, np.float32)


def descriptors(inputs: Sequence, structural: bool) -> np.ndarray:
    if not structural:
        formulas = [x if isinstance(x, str) else x.formula for x in inputs]
        return featurize(formulas)[0]
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    formulas, spg, crys = [], [], []
    for s in inputs:
        formulas.append(s.composition.formula)
        try:
            sga = SpacegroupAnalyzer(s, symprec=0.1)
            spg.append(sga.get_space_group_number())
            crys.append(sga.get_crystal_system())
        except Exception:
            spg.append(1)
            crys.append("triclinic")
    return featurize(formulas, spg, crys)[0]


def prepare_task(task: str, ids: List[str], inputs: Sequence, backbone: str = "orb_v2") -> Dict:
    """Builds (or loads) every cached input of one MatBench task."""
    from .backbone import cached_embeddings
    os.makedirs(CACHE, exist_ok=True)
    structural = task not in COMPOSITION_ONLY
    dpath = os.path.join(CACHE, f"qfa_{task}_desc_v1.npy")   # same for every backbone
    if os.path.exists(dpath):
        D = np.load(dpath)
    else:
        D = descriptors(inputs, structural)
        np.save(dpath, D)
    # backbone="none": plan B if orb-models cannot be installed -> descriptor (formula + symmetry)
    # branch + quantum head only, i.e. "QFA-Net without the structure branch"
    out = {"ids": list(ids), "desc": D, "structural": structural and backbone != "none"}
    if not out["structural"]:
        return out
    mode = "pooled" if task in LARGE else "atoms"
    out["mode"] = mode
    out["emb"] = cached_embeddings(task, inputs, ids, CACHE, name=backbone, mode=mode)
    if mode == "atoms":
        gpath = os.path.join(CACHE, f"qfa_{task}_graph_v1.npz")
        if os.path.exists(gpath):
            z = np.load(gpath, allow_pickle=True)
            out["graph"] = {i: (z[i + "_s"], z[i + "_d"], z[i + "_r"]) for i in ids}
        else:
            g = {i: image_aware_edges(s) for i, s in zip(ids, inputs)}
            flat = {}
            for i, (s_, d_, r_) in g.items():
                flat[i + "_s"], flat[i + "_d"], flat[i + "_r"] = s_, d_, r_
            np.savez(gpath, **flat)
            out["graph"] = g
    return out


def collate(prep: Dict, rows: Sequence[int], desc: np.ndarray, device: str = "cpu") -> Dict:
    """desc: already selected + scaled descriptor matrix (all rows of the task)."""
    batch = {"desc": torch.tensor(desc[rows], dtype=torch.float32, device=device)}
    if not prep["structural"]:
        return batch
    ids = [prep["ids"][r] for r in rows]
    if prep["mode"] == "pooled":
        H = np.stack([prep["emb"][i] for i in ids], axis=1)            # (L+1, B, 512)
        batch["H"] = torch.tensor(H, dtype=torch.float32, device=device)
        return batch
    Hs, src, dst, dist, gidx = [], [], [], [], []
    off = 0
    for k, i in enumerate(ids):
        H = prep["emb"][i]                                             # (L+1, N, 256)
        s_, d_, r_ = prep["graph"][i]
        Hs.append(H)
        src.append(s_ + off)
        dst.append(d_ + off)
        dist.append(r_)
        gidx.append(np.full(H.shape[1], k))
        off += H.shape[1]
    batch.update({
        "H": torch.tensor(np.concatenate(Hs, axis=1), dtype=torch.float32, device=device),
        "src": torch.tensor(np.concatenate(src), dtype=torch.long, device=device),
        "dst": torch.tensor(np.concatenate(dst), dtype=torch.long, device=device),
        "dist": torch.tensor(np.concatenate(dist), dtype=torch.float32, device=device),
        "atom_graph": torch.tensor(np.concatenate(gidx), dtype=torch.long, device=device),
        "n_graphs": len(ids),
    })
    return batch
