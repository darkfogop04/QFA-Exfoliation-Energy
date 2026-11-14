#!/usr/bin/env python
"""Build the compact graph cache for one MatBench task (run once per task).

Usage: python experiments/build_graph_cache.py --task matbench_phonons [--jobs 4]
"""
import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import graph_cache
from src.data.dataset import MatBenchExfoliationDataset
from src.utils.config import get_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--config", default="configs/optimize.yaml")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cfg = get_config(args.config)
    cfg["data"]["task_name"] = args.task
    ds = MatBenchExfoliationDataset(cfg)
    t0 = time.perf_counter()
    cache = graph_cache.load_or_build(ds.task, args.task, ds.graph_cache_dir, ds.cutoff_radius,
                                      ds.max_neighbors, args.jobs)
    n = len(cache["ids"])
    print(f"{args.task}: {n} crystals, {int(cache['atom_ptr'][-1])} atoms, {int(cache['edge_ptr'][-1])} edges, "
          f"{time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
