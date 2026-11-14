"""
Results table for the exfoliation runs (results/qfa_exfoliation/<dataset>/fold<k>.json), one table per dataset:
every QFA variant + RF from the fold JSON, plus two trivial baselines computed here on the same saved split
(data/exfoliation/<dataset>/folds.json): Median predictor and Dummy (training mean). "Training" = every
material not in the test fold (train + validation), as in MatBench's Dummy.

    python experiments/exfoliation_results_table.py                  # prints and writes RESULTS_TABLE.md
    python experiments/exfoliation_results_table.py --datasets twodmatpedia
"""
import argparse
import gzip
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data", "exfoliation")
OUT = os.path.join(ROOT, "results", "qfa_exfoliation")
DEFAULT = ["jarvis_dft_3d", "jarvis_dft_2d", "twodmatpedia"]
NICE = {"jarvis_dft_3d": "JARVIS-DFT 3D (bulk)", "jarvis_dft_2d": "JARVIS-DFT 2D (monolayers)",
        "twodmatpedia": "2DMatPedia (monolayers)"}
LABEL = {"hybrid_pruned": "QFA hybrid_pruned", "mlp_residual": "QFA mlp_residual", "classical": "QFA classical",
         "hybrid": "QFA hybrid", "qt_hybrid_pruned": "QFA-Net main (QT + pruned)",
         "rf_descriptors": "RF on our descriptors",
         "matminer_rf": "matminer RF (open-source, JARVIS-Leaderboard #2 recipe)",
         "matminer_xgboost": "matminer XGBoost (open-source, JARVIS-Leaderboard #1 recipe)","median": "Median predictor",
         "dummy": "Dummy (predicts the training mean)"}


def targets(name):
    with gzip.open(os.path.join(DATA, name, "dataset.json.gz"), "rt", encoding="utf-8") as f:
        entries = json.load(f)
    y = {}
    for e in entries:                       # same de-duplication as run_qfa_exfoliation.load_entries
        y.setdefault(e["id"], float(e["target"]))
    return y


def table(name):
    saved = json.load(open(os.path.join(DATA, name, "folds.json")))
    yd = targets(name)
    y = np.array([yd[i] for i in saved["ids"]])
    folds = [np.array(f) for f in saved["folds"]]
    done = sorted(int(f[4:-5]) for f in os.listdir(os.path.join(OUT, name))
                  if f.startswith("fold") and f[4:-5].isdigit())
    if not done:
        return None
    scores, info = {}, []
    for k in done:
        res = json.load(open(os.path.join(OUT, name, f"fold{k}.json")))
        test = folds[k]
        train = np.setdiff1d(np.arange(len(y)), test)
        row = dict(res["test"])
        row["median"] = float(np.mean(np.abs(y[test] - np.median(y[train]))))
        row["dummy"] = float(np.mean(np.abs(y[test] - y[train].mean())))
        for tag in ("matminer_rf", "matminer_xgboost"):     # open-source baselines (run_matminer_rf.py)
            mm = os.path.join(OUT, name, f"{tag}.json")
            if os.path.exists(mm):
                v = json.load(open(mm))["test"].get(str(k))
                if v is not None:
                    row[tag] = v
        for m, v in row.items():
            scores.setdefault(m, {})[k] = v
        info.append(f"fold {k}: {len(test)} test, {res.get('seconds', 0) / 60:.0f} min training")
    order = sorted(scores, key=lambda m: np.mean(list(scores[m].values())))
    head = "| Model | " + " | ".join(f"Fold {k}" for k in done) + (" | Mean ± std |" if len(done) > 1 else " |")
    lines = [f"### {NICE.get(name, name)}: test MAE, meV/atom (lower is better)", "",
             "; ".join(info), "", head, "|---|" + "---|" * (len(done) + (len(done) > 1))]
    for m in order:
        v = [scores[m].get(k, np.nan) for k in done]
        cells = " | ".join(f"{x:.2f}" for x in v)
        mean = f" | {np.nanmean(v):.2f} ± {np.nanstd(v):.2f}" if len(done) > 1 else ""
        lines.append(f"| {LABEL.get(m, m)} | {cells}{mean} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DEFAULT))
    a = ap.parse_args()
    parts = ["# QFA-Net on exfoliation datasets", "",
             "Same split for every model (5 random folds, seed 42); validation = 10 % of the training part; "
             "test fold used once. Std is over folds (population).", ""]
    for n in a.datasets.split(","):
        t = table(n)
        if t:
            parts += [t, ""]
    text = "\n".join(parts)
    print(text)
    open(os.path.join(OUT, "RESULTS_TABLE.md"), "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
