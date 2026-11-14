"""
QFA-Net lab benchmark (NumPy only): descriptor branch + quantum head + pruning + Quantum-Train
on the official MatBench folds.

    python experiments/run_qfa_lab.py --task jdft2d --folds 0,1,2,3,4
    python experiments/run_qfa_lab.py --task jdft2d --summary

Results: results/qfa_lab/<task>/fold<k>.json (one file per fold; existing files are skipped).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.qfa.lab.data import UNITS, leaderboard_fold_scores, load_task  # noqa: E402
from src.qfa.lab.experiment import run_fold, run_qt_control  # noqa: E402

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "qfa_lab"))


def summary(task: str) -> None:
    d = os.path.join(OUT, task)
    files = sorted(f for f in os.listdir(d) if f.startswith("fold")) if os.path.isdir(d) else []
    res = [json.load(open(os.path.join(d, f))) for f in files]
    if not res:
        print("no results")
        return
    print(f"{task}: {len(res)} folds ({UNITS[task]})")
    for k in res[0]["test"]:
        v = [r["test"][k] for r in res]
        print(f"  {k:18s} {np.mean(v):9.4f} ± {np.std(v):.4f}   {[round(x, 3) for x in v]}")
    lb = leaderboard_fold_scores(task)
    for name, v in sorted(lb.items(), key=lambda kv: np.mean(kv[1])):
        print(f"  [LB] {name:24s} {np.mean(v):9.4f}   {[round(x, 3) for x in v[:len(res)]]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--qt-control", action="store_true", help="Quantum-Train with vs without the QNN")
    a = ap.parse_args()
    if a.summary:
        summary(a.task)
        return
    data = load_task(a.task)
    os.makedirs(os.path.join(OUT, a.task), exist_ok=True)
    for f in [int(x) for x in a.folds.split(",")]:
        path = os.path.join(OUT, a.task, f"{'qtcontrol_' if a.qt_control else ''}fold{f}.json")
        if os.path.exists(path):
            print(f"fold {f}: done, skipping")
            continue
        print(f"{a.task} fold {f}", flush=True)
        if a.qt_control:
            info = run_qt_control(data, f, seeds=a.seeds or 3, log=lambda m: print(m, flush=True))
        else:
            info = run_fold(data, f, seeds=a.seeds, log=lambda m: print(m, flush=True))
        with open(path, "w") as fh:
            json.dump(info, fh, indent=1, default=float)
        print(f"fold {f}: {info['test']}", flush=True)


if __name__ == "__main__":
    main()
