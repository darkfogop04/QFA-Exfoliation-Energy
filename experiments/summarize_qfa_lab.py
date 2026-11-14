"""Tables for the QFA-Net lab results: per-variant means, paired fold wins, leaderboard comparison.

    python experiments/summarize_qfa_lab.py --task jdft2d [--json out.json]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.qfa.lab.data import UNITS, leaderboard_fold_scores  # noqa: E402

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "qfa_lab"))


def collect(task):
    d = os.path.join(OUT, task)
    folds = sorted(int(f[4:-5]) for f in os.listdir(d) if f.startswith("fold") and f.endswith(".json"))
    res = {f: json.load(open(os.path.join(d, f"fold{f}.json"))) for f in folds}
    return folds, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    folds, res = collect(a.task)
    reg = UNITS[a.task] != "ROC-AUC"
    out = {"task": a.task, "folds": folds, "unit": UNITS[a.task], "variants": {}, "leaderboard": {}, "paired": {}}
    variants = list(res[folds[0]]["test"].keys())
    for v in variants:
        vals = [res[f]["test"][v] for f in folds]
        out["variants"][v] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "folds": vals}
    for name, vals in leaderboard_fold_scores(a.task).items():
        sel = [vals[f] for f in folds]
        out["leaderboard"][name] = {"mean": float(np.mean(sel)), "folds": sel}

    def wins(x, y):  # folds where x is better than y
        return int(sum((res[f]["test"][x] < res[f]["test"][y]) if reg else (res[f]["test"][x] > res[f]["test"][y])
                       for f in folds))
    for x, y in [("hybrid", "classical"), ("hybrid_pruned", "classical"), ("hybrid", "mlp_residual"),
                 ("hybrid_pruned", "mlp_residual"), ("hybrid_pruned", "hybrid"), ("qt_hybrid_pruned", "hybrid_pruned"),
                 ("qt_hybrid_pruned", "qt_classical"), ("classical", "rf")]:
        if x in variants and y in variants:
            out["paired"][f"{x} vs {y}"] = f"{wins(x, y)}/{len(folds)}"
    # cost
    seeds = [s for f in folds for s in res[f]["per_seed"]]
    out["cost"] = {
        "classical_params": int(np.mean([s["classical_params"] for s in seeds])),
        "qhead_params_dense": int(np.mean([s["qhead_params_dense"] for s in seeds])),
        "qhead_params_pruned": float(np.mean([s["qhead_params_pruned"] for s in seeds])),
        "mlp_head_params": int(np.mean([s["mlp_head_params"] for s in seeds])),
        "angles_after_pruning": float(np.mean([s["gates_pruned"]["trainable"] for s in seeds])),
        "gates_dense": float(np.mean([s["gates_dense"]["total"] for s in seeds])),
        "gates_pruned": float(np.mean([s["gates_pruned"]["total"] for s in seeds])),
        "qt_trained": float(np.mean([s["qt_hybrid_trained"] for s in seeds if "qt_hybrid_trained" in s] or [0])),
        "qt_generated": float(np.mean([s["qt_generated"] for s in seeds if "qt_generated" in s] or [0])),
        "seeds_per_fold": res[folds[0]]["seeds"],
        "seconds_per_fold": float(np.mean([res[f]["seconds"] for f in folds])),
    }
    print(json.dumps(out, indent=1))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
