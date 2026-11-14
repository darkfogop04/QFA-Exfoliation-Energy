"""
Time-boxed QFA-Net run over the exfoliation datasets (same settings as run_final.py: seeds and stage caps per tier).

    python experiments/run_exfoliation_all.py --hours 6
    python experiments/run_exfoliation_all.py --hours 6 --datasets jarvis_dft_3d,twodmatpedia
    python experiments/run_exfoliation_all.py --dry-run

Order: fold 0 of every dataset, then folds 1-4 round-robin. Each fold runs experiments/run_qfa_exfoliation.py
in its own process; finished folds (results/qfa_exfoliation/<dataset>/fold<k>.json) are skipped, so rerunning
continues. A fold is skipped if its expected time (last measured fold of the same dataset, or a first guess)
does not fit before the deadline. Progress: results/qfa_exfoliation/STATUS.md.
matbench_jdft2d is left out by default: the same model on the same official folds is in results/qfa/jdft2d.
"""
import argparse
import datetime as dt
import gzip
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data", "exfoliation")
OUT = os.path.join(ROOT, "results", "qfa_exfoliation")
# exfoliation energy (meV/atom) only; the binding-energy sets (meV/A^2: massive_discovery_2025, mc2d_*) are not used
DEFAULT = ["jarvis_dft_3d", "jarvis_dft_2d", "twodmatpedia"]
STAGE_CAP = {"tiny": 8, "small": 12, "medium": 15, "large": 30}     # as in run_final.py
SEEDS = {"tiny": 3, "small": 2, "medium": 1, "large": 1}
GUESS_MIN = {"tiny": 45, "small": 70, "medium": 90, "large": 200}   # first guess incl. one-time ORB pass


def n_entries(name):
    with gzip.open(os.path.join(DATA, name, "dataset.json.gz"), "rt", encoding="utf-8") as f:
        return len({e["id"] for e in json.load(f)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=6)
    ap.add_argument("--datasets", default=",".join(DEFAULT))
    ap.add_argument("--folds", default="0,1,2,3,4", help="e.g. 0 for fold 0 of every dataset only")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from src.qfa.lab.experiment import policy
    names = a.datasets.split(",")
    tiers = {n: policy(int(0.8 * n_entries(n)))["tier"] for n in names}
    steps = [(n, int(f)) for f in a.folds.split(",") for n in names]
    started = dt.datetime.now()
    deadline = started + dt.timedelta(hours=a.hours)
    os.makedirs(OUT, exist_ok=True)
    measured, rows = {}, []

    def status():
        lines = ["# QFA-Net exfoliation datasets: run status", "",
                 f"Started {started:%Y-%m-%d %H:%M}, deadline {deadline:%Y-%m-%d %H:%M}, "
                 f"updated {dt.datetime.now():%Y-%m-%d %H:%M}", "",
                 "| Dataset | Tier | Fold | State | Minutes | Note |", "|---|---|---|---|---|---|"]
        lines += [f"| {r[0]} | {tiers[r[0]]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |" for r in rows]
        open(os.path.join(OUT, "STATUS.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    for n, f in steps:
        tier = tiers[n]
        if os.path.exists(os.path.join(OUT, n, f"fold{f}.json")):
            rows.append((n, f, "done earlier", "", ""))
            continue
        left = (deadline - dt.datetime.now()).total_seconds() / 60
        need = measured.get(n, GUESS_MIN[tier])
        if a.dry_run:
            print(f"{n:24s} fold {f} tier {tier:6s} seeds {SEEDS[tier]} cap {STAGE_CAP[tier]} min  ~{need} min")
            continue
        if need > left:
            rows.append((n, f, "skipped (time)", "", f"needs ~{need:.0f} min, {left:.0f} left"))
            status()
            continue
        print(f"\n=== {n} fold {f} ({tier}, {left:.0f} min left) ===", flush=True)
        t0 = time.time()
        cmd = [sys.executable, os.path.join(HERE, "run_qfa_exfoliation.py"), "--dataset", n, "--folds", str(f),
               "--seeds", str(SEEDS[tier]), "--max-minutes", str(STAGE_CAP[tier])]
        code = subprocess.call(cmd)
        mins = (time.time() - t0) / 60
        ok = code == 0 and os.path.exists(os.path.join(OUT, n, f"fold{f}.json"))
        if ok:
            measured[n] = mins
        rows.append((n, f, "done" if ok else "FAILED", f"{mins:.0f}", "" if ok else f"exit code {code}"))
        status()
    if a.dry_run:
        return
    status()
    print(f"\nFinished. See {os.path.join(OUT, 'STATUS.md')}")


if __name__ == "__main__":
    main()
