"""
Train and test QFA-Net on any exfoliation dataset made by experiments/download_exfoliation_data.py.

    python experiments/run_qfa_exfoliation.py --list
    python experiments/run_qfa_exfoliation.py --dataset jarvis_dft_3d --folds 0 --quick     # smoke test
    python experiments/run_qfa_exfoliation.py --dataset jarvis_dft_3d                       # 5 folds
    python experiments/run_qfa_exfoliation.py --dataset twodmatpedia --max-n 3000 --folds 0
    python experiments/run_qfa_exfoliation.py --dataset jarvis_dft_3d --summary

Folds: matbench_jdft2d uses the official MatBench folds; every other dataset gets 5 random folds
(seed 42, saved to data/exfoliation/<dataset>/folds.json so every run and model uses the same split).
Inside each fold run_fold keeps 10 % of the training part for validation; the test part is used once.
Results: results/qfa_exfoliation/<dataset>/fold<k>.json (finished folds are skipped, so runs resume).
Target unit: meV/atom; score: MAE (lower is better).
"""
import argparse
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
DATA = os.path.abspath(os.path.join(HERE, "..", "data", "exfoliation"))
META = os.path.abspath(os.path.join(HERE, "..", "data", "matbench_meta"))
OUT = os.path.abspath(os.path.join(HERE, "..", "results", "qfa_exfoliation"))


def available():
    if not os.path.isdir(DATA):
        return []
    return sorted(d for d in os.listdir(DATA) if os.path.exists(os.path.join(DATA, d, "dataset.json.gz")))


def make_folds(n: int, k: int = 5, seed: int = 42):
    idx = np.random.default_rng(seed).permutation(n)
    return [np.sort(part) for part in np.array_split(idx, k)]


def load_entries(name: str, max_n=None, seed: int = 42):
    path = os.path.join(DATA, name, "dataset.json.gz")
    if not os.path.exists(path):
        sys.exit(f"{path} not found: run experiments/download_exfoliation_data.py --only {name} first")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        entries = json.load(f)
    # drop duplicate ids, keep file order
    seen, uniq = set(), []
    for e in entries:
        if e["id"] not in seen:
            seen.add(e["id"])
            uniq.append(e)
    if max_n and len(uniq) > max_n:
        keep = np.sort(np.random.default_rng(seed).choice(len(uniq), max_n, replace=False))
        uniq = [uniq[i] for i in keep]
    return uniq


def folds_for(name: str, ids, tag: str):
    if name == "matbench_jdft2d":
        off = json.load(open(os.path.join(META, "folds.json")))["matbench_jdft2d"]
        pos = {m: i for i, m in enumerate(ids)}
        if all(m in pos for f in off for m in f):
            return [np.array(sorted(pos[m] for m in f)) for f in off]
        print("official fold ids not found in this file; using random folds instead")
    path = os.path.join(DATA, name, f"folds{tag}.json")
    if os.path.exists(path):
        saved = json.load(open(path))
        if saved["ids"] == list(ids):
            return [np.array(f) for f in saved["folds"]]
    folds = make_folds(len(ids))
    json.dump({"ids": list(ids), "folds": [f.tolist() for f in folds]}, open(path, "w"))
    return folds


def summary(name: str) -> None:
    d = os.path.join(OUT, name)
    files = sorted(f for f in os.listdir(d) if f.startswith("fold") and f[4:].split(".")[0].isdigit()) \
        if os.path.isdir(d) else []
    if not files:
        print(f"{name}: no results yet")
        return
    res = [json.load(open(os.path.join(d, f))) for f in files]
    print(f"\n{name}: MAE in meV/atom (lower is better), {len(res)} fold(s): {[f[:-5] for f in files]}")
    for k in res[0]["test"]:
        v = [r["test"][k] for r in res]
        print(f"  {k:22s} {np.mean(v):9.3f}  {[round(x, 3) for x in v]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="folder name under data/exfoliation (see --list)")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-n", type=int, default=None, help="random subset size for very large datasets")
    ap.add_argument("--max-minutes", type=float, default=None, help="cap per training stage")
    ap.add_argument("--backbone", default="orb_v2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.list or not a.dataset:
        print("datasets with labels (dataset.json.gz):", available() or "none yet")
        return
    if a.summary:
        summary(a.dataset)
        return

    from pymatgen.core import Structure
    from src.qfa.data import prepare_task
    from src.qfa.train import run_fold

    entries = load_entries(a.dataset, a.max_n)
    ids = [e["id"] for e in entries]
    inputs = [Structure.from_dict(e["structure"]) for e in entries]
    y = np.array([float(e["target"]) for e in entries])
    tag = f"_n{a.max_n}" if a.max_n else ""
    folds = folds_for(a.dataset, ids, tag)
    task = f"exf_{a.dataset}{tag}"          # cache key: data/cache/qfa_<task>_*
    print(f"{a.dataset}: {len(ids)} materials, target {y.min():.1f} to {y.max():.1f} meV/atom "
          f"(mean {y.mean():.1f}); fold sizes {[len(f) for f in folds]}")
    prep = prepare_task(task, ids, inputs, backbone=a.backbone)
    out = os.path.join(OUT, a.dataset + tag)
    os.makedirs(out, exist_ok=True)
    for f in [int(x) for x in a.folds.split(",")]:
        path = os.path.join(out, f"fold{f}{'_quick' if a.quick else ''}.json")
        if os.path.exists(path):
            print(f"fold {f}: done")
            continue
        print(f"fold {f}", flush=True)
        info = run_fold(task, prep, prep["desc"], y, folds[f], seeds=a.seeds, quick=a.quick,
                        max_minutes=a.max_minutes, device=a.device, log=lambda m: print(m, flush=True))
        info["dataset"] = a.dataset
        json.dump(info, open(path, "w"), indent=1, default=float)
        print(f"fold {f}: {info['test']}", flush=True)
    summary(a.dataset + tag)


if __name__ == "__main__":
    main()
