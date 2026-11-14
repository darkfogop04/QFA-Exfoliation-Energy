import logging
import torch
from torch_geometric.data import Data
from typing import Optional

from src.data.featurizers import AtomFeaturizer, BondFeaturizer

logger = logging.getLogger(__name__)


def compact_edges(structure, cutoff_radius: float = 8.0, max_neighbors: int = 12):
    """Neighbour list of a crystal as (src, dst, distance) lists.

    For every atom: up to `max_neighbors` nearest neighbours within
    `cutoff_radius`, added in both directions; repeated (src, dst) pairs keep
    the first distance. Crystals with no neighbours get self-loops (distance 0).
    """
    n_atoms = len(structure)
    src_indices, dst_indices, distances = [], [], []
    for i, neighbors in enumerate(structure.get_all_neighbors(cutoff_radius)):
        for n in sorted(neighbors, key=lambda n: n.nn_distance)[:max_neighbors]:
            j, distance = n.index, n.nn_distance
            src_indices += [i, j]
            dst_indices += [j, i]
            distances += [distance, distance]

    if len(src_indices) == 0:
        logger.warning(f"No neighbors found within cutoff {cutoff_radius} for structure. Adding self-loops.")
        src_indices = list(range(n_atoms))
        dst_indices = list(range(n_atoms))
        distances = [0.0] * n_atoms

    # Remove duplicate (src, dst) pairs created by adding j -> i manually
    edges, unique_src, unique_dst, unique_dist = set(), [], [], []
    for s, d, dist in zip(src_indices, dst_indices, distances):
        if (s, d) not in edges:
            edges.add((s, d))
            unique_src.append(s)
            unique_dst.append(d)
            unique_dist.append(dist)
    return unique_src, unique_dst, unique_dist

class CrystalGraphConverter:
    """
    Converts a pymatgen Structure into a PyTorch Geometric Data object.
    """
    def __init__(self, cutoff_radius: float = 8.0, max_neighbors: int = 12):
        self.cutoff_radius = cutoff_radius
        self.max_neighbors = max_neighbors
        self.atom_featurizer = AtomFeaturizer()
        self.bond_featurizer = BondFeaturizer(cutoff_radius=cutoff_radius)
        
    def convert(self, structure, target: Optional[float] = None) -> Data:
        """
        Converts the structure to a PyG Data object.
        
        Args:
            structure (pymatgen.core.Structure): The input crystal structure.
            target (float, optional): The target property value.
            
        Returns:
            torch_geometric.data.Data: PyG Data object containing x, edge_index, edge_attr, y.
        """
        # Node features
        x = self.atom_featurizer(structure)
        n_atoms = len(structure)
        
        unique_src, unique_dst, unique_dist = compact_edges(structure, self.cutoff_radius, self.max_neighbors)
        edge_index = torch.tensor([unique_src, unique_dst], dtype=torch.long)
        unique_dist_tensor = torch.tensor(unique_dist, dtype=torch.float32)
        
        # Edge features
        edge_attr = self.bond_featurizer(unique_dist_tensor)
        
        # Target
        y = torch.tensor([target], dtype=torch.float32) if target is not None else None
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data
