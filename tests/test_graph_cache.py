"""Tests for the compact graph cache and the training time budget."""
import pytest
import torch


def _structures():
    from pymatgen.core import Lattice, Structure
    return [
        Structure(Lattice.cubic(5.0), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]]),
        Structure(Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
        Structure(Lattice.hexagonal(3.2, 12.0), ["Mo", "S", "S"],
                  [[1/3, 2/3, 0.5], [2/3, 1/3, 0.62], [2/3, 1/3, 0.38]]),
        Structure(Lattice.cubic(30.0), ["H"], [[0, 0, 0]]),        # no neighbours → self-loop
        Structure(Lattice.cubic(4.0), ["Pu", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]),  # Z > 92 → zero row
    ]


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_cache_matches_converter(n_jobs):
    """Graphs from the cache are identical to CrystalGraphConverter's output."""
    from src.data.crystal_graph import CrystalGraphConverter
    from src.data.graph_cache import GraphSubset, build_cache, positions_for

    structs = _structures()
    targets = [1.5, -2.0, 30.0, 0.0, 7.25]
    ids = [f"mb-test-{i}" for i in range(len(structs))]
    cache = build_cache(ids, structs, targets, cutoff=8.0, max_neighbors=12, n_jobs=n_jobs)
    mean, std = 3.0, 2.0
    ds = GraphSubset(cache, positions_for(cache, ids), mean, std)
    conv = CrystalGraphConverter(cutoff_radius=8.0, max_neighbors=12)
    assert len(ds) == len(structs)
    for k, (s, t) in enumerate(zip(structs, targets)):
        ref = conv.convert(s, t)
        got = ds[k]
        assert torch.equal(got.x, ref.x)
        assert torch.equal(got.edge_index, ref.edge_index)
        torch.testing.assert_close(got.edge_attr, ref.edge_attr)
        torch.testing.assert_close(got.y, (ref.y - mean) / std)


def test_positions_follow_ids():
    from src.data.graph_cache import build_cache, positions_for
    structs = _structures()[:3]
    cache = build_cache(["a", "b", "c"], structs, [1, 2, 3])
    assert positions_for(cache, ["c", "a"]) == [2, 0]


def test_trainer_time_budget():
    """A tiny time budget stops training after one epoch and restores the best weights."""
    from torch_geometric.loader import DataLoader
    from src.models.classical_baseline import ClassicalGNN
    from src.training.trainer import Trainer
    from tests.test_smoke import _get_smoke_config, _make_synthetic_dataset

    cfg = _get_smoke_config()
    cfg["training"]["epochs"] = 50
    cfg["training"]["max_minutes"] = 1e-6
    model = ClassicalGNN(cfg)
    data = _make_synthetic_dataset(8)
    loader = DataLoader(data, batch_size=4)
    trainer = Trainer(model, cfg, device="cpu")
    res = trainer.train(loader, val_loader=loader)
    assert trainer.stopped_by_time
    assert len(res["train_history"]) == 1
