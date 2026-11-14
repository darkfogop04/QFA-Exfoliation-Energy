"""
MatBench data for the lab runs: targets, compositions and symmetry labels for all 13
tasks (data/matbench_meta/*.json.bz2, from the matbench repository's artifacts), and
the official test folds (data/matbench_meta/folds.json, read from the test-set ids
of the published leaderboard submissions, so the folds are exactly MatBench's).

No crystal structures are available on this machine, so the lab uses composition
(Magpie, 132) + symmetry (8) features. The full QFA-Net (structure branch) runs with
PyTorch on a local machine: see experiments/run_qfa.py.
"""
import bz2
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from .featurize import featurize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
META = os.path.join(ROOT, "data", "matbench_meta")
CACHE = os.path.join(ROOT, "data", "cache")

TASKS = ["steels", "jdft2d", "phonons", "expt_gap", "dielectric", "expt_is_metal", "glass",
         "log_gvrh", "log_kvrh", "perovskites", "mp_gap", "mp_is_metal", "mp_e_form"]
CLASSIFICATION = {"expt_is_metal", "glass", "mp_is_metal"}
COMPOSITION_ONLY = {"steels", "expt_gap", "expt_is_metal", "glass"}
UNITS = {"steels": "MPa", "jdft2d": "meV/atom", "phonons": "cm^-1", "expt_gap": "eV",
         "dielectric": "unitless", "log_gvrh": "log10 GPa", "log_kvrh": "log10 GPa",
         "perovskites": "eV/unit cell", "mp_gap": "eV", "mp_e_form": "eV/atom",
         "expt_is_metal": "ROC-AUC", "glass": "ROC-AUC", "mp_is_metal": "ROC-AUC"}


def task_type(task: str) -> str:
    return "classification" if task in CLASSIFICATION else "regression"


def load_task(task: str) -> Dict:
    """Returns dict with ids, X, names, y, folds (list of 5 test-index arrays)."""
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, f"qfa_lab_{task}_v1.npz")
    raw = json.load(bz2.open(os.path.join(META, f"matbench_{task}.json.bz2")))
    ids = [raw["mbid"][str(i)] for i in range(len(raw["mbid"]))]
    target_key = [k for k in raw if k not in ("mbid", "composition", "spg_num", "crys_sys")][0]
    y = np.array([float(raw[target_key][str(i)]) for i in range(len(ids))])
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        X, names = z["X"], list(z["names"])
    else:
        comps = [raw["composition"][str(i)] for i in range(len(ids))]
        if "spg_num" in raw:
            spg = [int(raw["spg_num"][str(i)]) for i in range(len(ids))]
            crys = [raw["crys_sys"][str(i)] for i in range(len(ids))]
            X, names = featurize(comps, spg, crys)
        else:
            X, names = featurize(comps)
        np.savez_compressed(cache, X=X, names=np.array(names))
    folds_ids = json.load(open(os.path.join(META, "folds.json")))[f"matbench_{task}"]
    pos = {m: i for i, m in enumerate(ids)}
    folds = [np.array(sorted(pos[m] for m in f)) for f in folds_ids]
    return {"task": task, "ids": ids, "X": X, "names": names, "y": y, "folds": folds,
            "type": task_type(task), "target": target_key}


def leaderboard_fold_scores(task: str) -> Dict[str, List[float]]:
    d = json.load(open(os.path.join(META, "leaderboard_fold_scores.json")))
    return d.get(f"matbench_{task}", {})


def split_train_val(n_train_idx: np.ndarray, seed: int, val_fraction: float = 0.1
                    ) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(n_train_idx)
    n_val = max(1, int(round(val_fraction * len(perm))))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])
