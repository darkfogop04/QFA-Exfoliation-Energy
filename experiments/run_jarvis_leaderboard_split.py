"""
QFA-Net on the official JARVIS-Leaderboard split of dft_3d exfoliation energy, so its score can be put next to the
leaderboard models (https://pages.nist.gov/jarvis_leaderboard/AI/SinglePropertyPrediction/dft_3d_exfoliation_energy).

Split file: data/exfoliation/jarvis_dft_3d/jarvis_leaderboard_split.json (copy of
jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction/dft_3d_exfoliation_energy.json.zip): train 650, val 81,
test 81. Only these 812 materials are used (our file has one extra, JVASP-30404, which is left out). As in every
other run, QFA trains on train+val (731) with its own 10 % internal validation, and the 81 test materials are used
once. Median predictor and Dummy (train+val mean) are computed on the same split.

    python experiments/run_jarvis_leaderboard_split.py            # ~45 min (first run also builds the ORB cache)
    python experiments/run_jarvis_leaderboard_split.py --quick    # smoke test
"""
import argparse
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data", "exfoliation", "jarvis_dft_3d")
OUT = os.path.join(ROOT, "results", "qfa_exfoliation", "jarvis_dft_3d_leaderboard")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)          # tiny tier, same as run_exfoliation_all.py
    ap.add_argument("--max-minutes", type=float, default=8)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    from pymatgen.core import Structure
    from src.qfa.data import prepare_task
    from src.qfa.train import run_fold

    split = json.load(open(os.path.join(DATA, "jarvis_leaderboard_split.json")))
    with gzip.open(os.path.join(DATA, "dataset.json.gz"), "rt", encoding="utf-8") as f:
        entries = {}
        for e in json.load(f):
            entries.setdefault(e["id"], e)
    ids = list(split["train"]) + list(split["val"]) + list(split["test"])
    for i in ids:                                  # targets must agree with the leaderboard file
        ref = {**split["train"], **split["val"], **split["test"]}[i]
        assert abs(float(entries[i]["target"]) - ref) < 1e-6, i
    y = np.array([float(entries[i]["target"]) for i in ids])
    test_idx = np.arange(len(ids) - len(split["test"]), len(ids))
    print(f"leaderboard split: {len(ids)} materials, test {len(test_idx)}", flush=True)

    task = "exf_jarvis_dft_3d_lb"                 # own cache key (row order differs from the 5-fold runs)
    prep = prepare_task(task, ids, [Structure.from_dict(entries[i]["structure"]) for i in ids])
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "test_quick.json" if a.quick else "test.json")
    info = run_fold(task, prep, prep["desc"], y, test_idx, seeds=a.seeds, quick=a.quick,
                    max_minutes=a.max_minutes, log=lambda m: print(m, flush=True))
    tr = y[: test_idx[0]]
    yt = y[test_idx]
    info["test"]["median"] = float(np.mean(np.abs(yt - np.median(tr))))
    info["test"]["dummy"] = float(np.mean(np.abs(yt - tr.mean())))
    info["dataset"] = "jarvis_dft_3d (JARVIS-Leaderboard split)"
    json.dump(info, open(path, "w"), indent=1, default=float)
    for k, v in sorted(info["test"].items(), key=lambda kv: kv[1]):
        print(f"  {k:20s} {v:8.3f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
