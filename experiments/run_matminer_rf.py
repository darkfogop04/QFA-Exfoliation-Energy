"""
Open-source baseline "matminer_rf" (JARVIS-Leaderboard, team UofT, 2023) re-run on our exfoliation datasets.

Recipe copied from jarvis_leaderboard/contributions/matminer_rf/run.py:
  features: SiteStatsFingerprint(CoordinationNumber_ward-prb-2017, LocalPropertyDifference_ward-prb-2017),
            StructuralHeterogeneity, MaximumPackingEfficiency, ChemicalOrdering (structure) +
            Stoichiometry, ElementProperty(magpie), ValenceOrbital(frac), IonProperty(fast) (composition);
            sites moved into the unit cell first; failed features left NaN (imputed)
  model:    SimpleImputer -> StandardScaler -> RandomForestRegressor(n_estimators=100, max_features=1/3,
            bootstrap=False, random_state=0)
Evaluations:
  1. JARVIS-Leaderboard split of dft_3d exfoliation (fit on its 650 train only, as the recipe does; 81 test).
     Published score: 42.7554 meV/atom -> checks that our re-run matches.
  2. Our 5 random folds (data/exfoliation/<dataset>/folds.json) of every dataset; fit on everything outside the
     test fold, the same data the QFA models train on.
Features are cached in data/cache/mmrf_<dataset>_X.npz. Results: results/qfa_exfoliation/<dataset>/matminer_rf.json.

    python experiments/run_matminer_rf.py --n-jobs 4                 # matminer_rf
    python experiments/run_matminer_rf.py --n-jobs 4 --model xgb     # matminer_xgboost (leaderboard #1)
"""
import argparse
import gzip
import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data", "exfoliation")
CACHE = os.path.join(ROOT, "data", "cache")
OUT = os.path.join(ROOT, "results", "qfa_exfoliation")
DATASETS = ["jarvis_dft_3d", "jarvis_dft_2d", "twodmatpedia"]


def load(name):
    from pymatgen.core import Structure
    with gzip.open(os.path.join(DATA, name, "dataset.json.gz"), "rt", encoding="utf-8") as f:
        entries = {}
        for e in json.load(f):
            entries.setdefault(e["id"], e)
    return entries, Structure


def featurize(name, n_jobs):
    path = os.path.join(CACHE, f"mmrf_{name}_X.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return list(z["ids"]), z["X"]
    import pandas as pd
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital, IonProperty
    from matminer.featurizers.structure import (SiteStatsFingerprint, StructuralHeterogeneity, ChemicalOrdering,
                                                StructureComposition, MaximumPackingEfficiency)
    entries, Structure = load(name)
    ids = list(entries)
    structs = []
    for i in ids:
        s = Structure.from_dict(entries[i]["structure"])
        for site in s.sites:
            site.to_unit_cell(in_place=True)
        structs.append(s)
    df = pd.DataFrame({"structure": structs}, index=ids)
    feat = MultipleFeaturizer([
        SiteStatsFingerprint.from_preset("CoordinationNumber_ward-prb-2017"),
        SiteStatsFingerprint.from_preset("LocalPropertyDifference_ward-prb-2017"),
        StructuralHeterogeneity(), MaximumPackingEfficiency(), ChemicalOrdering(),
        StructureComposition(Stoichiometry()), StructureComposition(ElementProperty.from_preset("magpie")),
        StructureComposition(ValenceOrbital(props=["frac"])), StructureComposition(IonProperty(fast=True))])
    feat.set_n_jobs(n_jobs)
    feat.set_chunksize(20)
    t0 = time.time()
    X = feat.featurize_dataframe(df, "structure", ignore_errors=True, pbar=False).drop(columns="structure")
    X = X.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    failed = int(np.isnan(X).any(axis=1).sum())
    print(f"{name}: {X.shape[1]} features for {len(ids)} materials, {failed} with some failed features, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, ids=np.array(ids, dtype=object), X=X)
    return ids, X


MODELS = {"rf": ("matminer_rf", 42.7554), "xgb": ("matminer_xgboost", 40.887)}   # published leaderboard MAE


def model(n_jobs, kind="rf"):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "xgb":                     # jarvis_leaderboard/contributions/matminer_xgboost/run.py (CPU "hist")
        import xgboost as xgb
        m = xgb.XGBRegressor(n_estimators=10000, learning_rate=0.1, reg_lambda=0.01, reg_alpha=0.01,
                             colsample_bytree=0.3, colsample_bylevel=0.5, num_parallel_tree=8,
                             tree_method="hist", n_jobs=n_jobs)
    else:
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=100, max_features=1 / 3, n_jobs=n_jobs, bootstrap=False,
                                  random_state=0)
    return Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("model", m)])


def fit_mae(X, y, tr, te, n_jobs, kind="rf"):
    keep = ~np.all(np.isnan(X[tr]), axis=0)          # SimpleImputer drops all-NaN columns anyway
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.time()
        m = model(n_jobs, kind).fit(X[tr][:, keep], y[tr])
        p = m.predict(X[te][:, keep])
    return float(np.mean(np.abs(p - y[te]))), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--model", choices=list(MODELS), default="rf")
    a = ap.parse_args()
    for name in a.datasets.split(","):
        ids, X = featurize(name, a.n_jobs)
        entries, _ = load(name)
        pos = {i: k for k, i in enumerate(ids)}
        tag, published = MODELS[a.model]
        res = {"model": f"{tag} (JARVIS-Leaderboard recipe, UofT)", "dataset": name, "test": {}, "seconds": {}}
        if name == "jarvis_dft_3d":
            split = json.load(open(os.path.join(DATA, name, "jarvis_leaderboard_split.json")))
            tr = np.array([pos[i] for i in split["train"]])
            te = np.array([pos[i] for i in split["test"]])
            y = np.array([float(entries[i]["target"]) for i in ids])
            mae, sec = fit_mae(X, y, tr, te, a.n_jobs, a.model)
            res["leaderboard_split"] = {"mae": mae, "published_mae": published, "seconds": sec, "n_train": len(tr), "n_test": len(te)}
            print(f"{name} leaderboard split: MAE {mae:.3f} (published {published})  ({sec:.0f} s)", flush=True)
        saved = json.load(open(os.path.join(DATA, name, "folds.json")))
        y = np.array([float(entries[i]["target"]) for i in saved["ids"]])
        Xf = X[[pos[i] for i in saved["ids"]]]
        for k, fold in enumerate(saved["folds"]):
            te = np.array(fold)
            tr = np.setdiff1d(np.arange(len(y)), te)
            mae, sec = fit_mae(Xf, y, tr, te, a.n_jobs, a.model)
            res["test"][k], res["seconds"][k] = mae, sec
            print(f"{name} fold {k}: MAE {mae:.3f}  ({sec:.0f} s)", flush=True)
        json.dump(res, open(os.path.join(OUT, name, f"{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
