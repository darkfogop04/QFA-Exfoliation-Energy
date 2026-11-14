"""
Time-boxed final benchmark of QFA-Net on the MatBench tasks (laptop, CPU, no git needed).

    python experiments/run_final.py --hours 5.5            # the real run
    python experiments/run_final.py --hours 5.5 --dry-run  # only print the plan

Plan (in this order, skipping whatever would not finish before the deadline):
  phase 1  jdft2d (exfoliation energy), all 5 folds          <- the main result
  phase 2  fold 0 of every other feasible task (same folds as the leaderboard, so fold-0 scores
           are compared with the leaderboard's fold-0 scores)
  phase 3  folds 1-4 of the small tasks, round-robin, while time is left
mp_gap, mp_is_metal and mp_e_form (100k+ crystals) are only run with --include-large.

Everything is resumable: finished folds are saved as results/qfa/<task>/fold<k>.json and skipped
next time. Progress, timings and errors are written to results/qfa/STATUS.md after every fold,
and results/qfa/final_results.json is rewritten at the end (and after every fold).
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import run_qfa  # noqa: E402  (experiments/run_qfa.py)

OUT = os.path.join(ROOT, "results", "qfa")
STATUS = os.path.join(OUT, "STATUS.md")
EXPORT = os.path.join(OUT, "final_results.json")

MAIN = "jdft2d"
FOLD0_TASKS = ["steels", "phonons", "expt_gap", "glass", "expt_is_metal", "dielectric",
               "log_gvrh", "log_kvrh", "perovskites"]
SMALL_TASKS = ["steels", "phonons", "expt_gap", "glass", "expt_is_metal", "dielectric"]
LARGE_TASKS = ["mp_gap", "mp_is_metal", "mp_e_form"]

# first guess of minutes per fold (backbone pass + training) before anything was measured;
# replaced by the measured time of the previous fold of the same task
GUESS_MIN = {"steels": 10, "jdft2d": 25, "phonons": 40, "expt_gap": 45, "glass": 50, "expt_is_metal": 45,
             "dielectric": 70, "log_gvrh": 90, "log_kvrh": 90, "perovskites": 120,
             "mp_gap": 400, "mp_is_metal": 400, "mp_e_form": 450}
# cap (minutes) for each training stage, so a slow task cannot eat the whole budget
STAGE_CAP = {"tiny": 8, "small": 12, "medium": 15, "large": 30}
# seeds per fold (ensemble size); fewer than the full policy so that more tasks fit in the time box
SEEDS = {"tiny": 3, "small": 2, "medium": 1, "large": 1}


def plan(include_large: bool):
    steps = [(MAIN, f) for f in range(5)]
    steps += [(t, 0) for t in FOLD0_TASKS]
    if include_large:
        steps += [(t, 0) for t in LARGE_TASKS]
    for f in range(1, 5):
        steps += [(t, f) for t in SMALL_TASKS]
    return steps


def done_path(task, fold):
    return os.path.join(OUT, task, f"fold{fold}.json")


def write_status(log_rows, started, deadline):
    os.makedirs(OUT, exist_ok=True)
    lines = ["# QFA-Net final run status", "",
             f"Started: {started:%Y-%m-%d %H:%M}   Deadline: {deadline:%Y-%m-%d %H:%M}   "
             f"Updated: {dt.datetime.now():%Y-%m-%d %H:%M}", "",
             "| Task | Fold | State | Minutes | Main test score (qt_hybrid_pruned or hybrid_pruned) | Note |",
             "|---|---|---|---|---|---|"]
    for r in log_rows:
        lines.append(f"| {r['task']} | {r['fold']} | {r['state']} | {r.get('minutes', '')} | "
                     f"{r.get('score', '')} | {r.get('note', '')} |")
    lines += ["", "Results: results/qfa/<task>/fold<k>.json; all in one file: results/qfa/final_results.json.",
              "Rerun the same command to continue; finished folds are skipped."]
    with open(STATUS, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=5.5)
    ap.add_argument("--include-large", action="store_true")
    ap.add_argument("--backbone", default="orb_v2", help="orb_v2, or none = plan B without the structure branch")
    ap.add_argument("--seeds", type=int, default=None, help="override seeds per fold for every task")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    global OUT, STATUS, EXPORT
    if a.backbone == "none":   # plan B results are kept apart from the full-model results
        OUT = os.path.join(ROOT, "results", "qfa_formula_only")
        STATUS, EXPORT = os.path.join(OUT, "STATUS.md"), os.path.join(OUT, "final_results.json")
        run_qfa.OUT = OUT
    try:
        import torch
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    except Exception:  # noqa: BLE001
        pass
    started = dt.datetime.now()
    deadline = started + dt.timedelta(hours=a.hours)
    steps = plan(a.include_large)
    if a.dry_run:
        for t, f in steps:
            print(f"{t:14s} fold {f}  {'done' if os.path.exists(done_path(t, f)) else 'todo'}"
                  f"  (~{GUESS_MIN[t]} min first guess)")
        return
    measured, prepared, failed, rows = {}, {}, set(), []
    for task, fold in steps:
        if task in failed:
            continue
        if os.path.exists(done_path(task, fold)):
            rows.append({"task": task, "fold": fold, "state": "done earlier"})
            continue
        left = (deadline - dt.datetime.now()).total_seconds() / 60
        need = measured.get(task, GUESS_MIN[task])
        if need > left:
            rows.append({"task": task, "fold": fold, "state": "skipped (time)",
                         "note": f"needs ~{need:.0f} min, {left:.0f} left"})
            write_status(rows, started, deadline)
            continue
        t0 = time.time()
        print(f"\n=== {task} fold {fold}  ({left:.0f} min left) ===", flush=True)
        try:
            if task not in prepared:
                prepared.clear()          # keep one task in memory at a time (caches reload fast)
                ids, inputs, y, folds = run_qfa.load_matbench(task)
                from src.qfa.data import prepare_task
                prepared[task] = (prepare_task(task, ids, inputs, backbone=a.backbone), y, folds)
            prep, y, folds = prepared[task]
            from src.qfa.lab.experiment import policy
            from src.qfa.train import run_fold
            import numpy as np
            tier = policy(len(y) - len(folds[fold]))["tier"]
            info = run_fold(task, prep, prep["desc"], y, folds[fold], max_minutes=STAGE_CAP[tier],
                            seeds=a.seeds or SEEDS[tier],
                            log=lambda m: print(m, flush=True))
            os.makedirs(os.path.join(OUT, task), exist_ok=True)
            with open(done_path(task, fold), "w") as fh:
                json.dump(info, fh, indent=1, default=float)
            minutes = (time.time() - t0) / 60
            measured[task] = minutes
            main_key = "qt_hybrid_pruned" if "qt_hybrid_pruned" in info["test"] else "hybrid_pruned"
            rows.append({"task": task, "fold": fold, "state": "done", "minutes": f"{minutes:.0f}",
                         "score": f"{info['test'][main_key]:.4f} ({main_key})"})
            print(f"{task} fold {fold}: {info['test']}  [{minutes:.0f} min]", flush=True)
        except Exception as e:  # noqa: BLE001  keep going with the next task
            traceback.print_exc()
            failed.add(task)
            rows.append({"task": task, "fold": fold, "state": "FAILED",
                         "minutes": f"{(time.time() - t0) / 60:.0f}", "note": f"{type(e).__name__}: {e}"[:200]})
        write_status(rows, started, deadline)
        try:
            run_qfa.export(run_qfa.TASKS, EXPORT)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
    write_status(rows, started, deadline)
    run_qfa.export(run_qfa.TASKS, EXPORT)
    print(f"\nFinished. See {STATUS} and {EXPORT}")


if __name__ == "__main__":
    main()
