"""
Compact, on-disk graph cache for any MatBench structure task.

Why: building neighbour lists with pymatgen is the slowest part of data loading
(and was repeated for every fold), and storing full feature matrices
(92-wide one-hot atoms, 41-wide Gaussian edges) for 100k+ crystals does not
fit in memory. The cache stores only

    atomic numbers (int16), local edge indices (int32), edge distances (float32)

for every crystal, built once in parallel. `GraphSubset` expands them to the
exact same `Data(x, edge_index, edge_attr, y)` that `CrystalGraphConverter`
produces, one crystal at a time, when the DataLoader asks for it.
"""
import logging
import os
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.data.crystal_graph import compact_edges
from src.data.featurizers import BondFeaturizer

logger = logging.getLogger(__name__)

NUM_ELEMENTS = 92
CACHE_VERSION = 1


def _compact_one(args):
    structure, cutoff, max_nb = args
    z = np.array([site.specie.Z for site in structure], dtype=np.int16)
    src, dst, dist = compact_edges(structure, cutoff, max_nb)
    return z, np.array([src, dst], dtype=np.int32), np.array(dist, dtype=np.float32)


def build_cache(ids: Sequence[str], structures: Sequence, targets: Sequence[float],
                cutoff: float = 8.0, max_neighbors: int = 12, n_jobs: int = 1) -> Dict:
    """Neighbour lists for every structure, concatenated into flat arrays."""
    work = [(s, cutoff, max_neighbors) for s in structures]
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            parts = pool.map(_compact_one, work, chunksize=64)
    else:
        parts = [_compact_one(w) for w in work]

    n_atoms = np.array([len(p[0]) for p in parts], dtype=np.int64)
    n_edges = np.array([p[1].shape[1] for p in parts], dtype=np.int64)
    return {
        "version": CACHE_VERSION, "cutoff": cutoff, "max_neighbors": max_neighbors,
        "ids": list(ids),
        "y": torch.tensor(np.asarray(targets, dtype=np.float64)),
        "z": torch.from_numpy(np.concatenate([p[0] for p in parts])),
        "edge_index": torch.from_numpy(np.concatenate([p[1] for p in parts], axis=1)),
        "dist": torch.from_numpy(np.concatenate([p[2] for p in parts])),
        "atom_ptr": torch.from_numpy(np.concatenate([[0], np.cumsum(n_atoms)])),
        "edge_ptr": torch.from_numpy(np.concatenate([[0], np.cumsum(n_edges)])),
    }


def cache_path(cache_dir: str, task_name: str, cutoff: float, max_neighbors: int) -> str:
    return os.path.join(cache_dir, f"{task_name}_c{cutoff:g}_n{max_neighbors}_v{CACHE_VERSION}.pt")


def load_or_build(task, task_name: str, cache_dir: str, cutoff: float, max_neighbors: int,
                  n_jobs: int = 1) -> Dict:
    """Load the cache for `task` or build it from the full dataset (fold 0 train + test)."""
    path = cache_path(cache_dir, task_name, cutoff, max_neighbors)
    if os.path.exists(path):
        logger.info(f"Loading graph cache {path}")
        return torch.load(path, weights_only=False)
    tr_x, tr_y = task.get_train_and_val_data(0)
    te_x, te_y = task.get_test_data(0, include_target=True)
    ids = list(tr_x.index) + list(te_x.index)
    structures = list(tr_x.values) + list(te_x.values)
    targets = list(tr_y.values) + list(te_y.values)
    logger.info(f"Building graph cache for {task_name}: {len(ids)} structures, {n_jobs} processes")
    cache = build_cache(ids, structures, targets, cutoff, max_neighbors, n_jobs)
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(cache, path)
    logger.info(f"Saved {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return cache


class GraphSubset(Dataset):
    """A list of crystals from the cache, expanded to full PyG graphs on access."""

    def __init__(self, cache: Dict, positions: Sequence[int], mean: float, std: float,
                 bond_featurizer: Optional[BondFeaturizer] = None):
        self.c = cache
        self.pos = list(positions)
        self.mean, self.std = mean, std
        self.bond = bond_featurizer or BondFeaturizer(cutoff_radius=cache["cutoff"])

    def __len__(self):
        return len(self.pos)

    def raw_targets(self) -> torch.Tensor:
        return self.c["y"][self.pos].to(torch.float32)

    def __getitem__(self, k):
        i = self.pos[k]
        a0, a1 = int(self.c["atom_ptr"][i]), int(self.c["atom_ptr"][i + 1])
        e0, e1 = int(self.c["edge_ptr"][i]), int(self.c["edge_ptr"][i + 1])
        z = self.c["z"][a0:a1].long() - 1                     # Z=1 → index 0 (as AtomFeaturizer)
        x = torch.zeros(a1 - a0, NUM_ELEMENTS)
        ok = (z >= 0) & (z < NUM_ELEMENTS)                     # out-of-range elements → all zeros
        x[torch.nonzero(ok).flatten(), z[ok]] = 1.0
        edge_index = self.c["edge_index"][:, e0:e1].long()
        edge_attr = self.bond(self.c["dist"][e0:e1])
        y = (torch.tensor([float(self.c["y"][i])], dtype=torch.float32) - self.mean) / self.std
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def positions_for(cache: Dict, ids: Sequence[str]) -> List[int]:
    lookup = cache.get("_lookup")
    if lookup is None:
        lookup = cache["_lookup"] = {mbid: i for i, mbid in enumerate(cache["ids"])}
    return [lookup[mbid] for mbid in ids]
