"""
Full QFA-Net on MatBench (PyTorch; run on your own machine).

Setup once (inside the project venv):
    pip install orb-models ase
    pip install --no-deps matbench==0.6

Examples:
    python experiments/run_qfa.py --task jdft2d --folds 0 --quick      # 5-minute smoke test
    python experiments/run_qfa.py --task jdft2d                        # 5 folds, full run
    python experiments/run_qfa.py --task all --folds 0                 # every task, fold 0
    python experiments/run_qfa.py --task jdft2d --summary              # table vs leaderboard

Results: results/qfa/<task>/fold<k>.json (existing files are skipped, so runs can resume).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.qfa.lab.data import TASKS, UNITS, leaderboard_fold_scores  # noqa: E402

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "qfa"))
META = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "matbench_meta"))


def load_matbench(task: str):
    from matbench.bench import MatbenchBenchmark
    mb = MatbenchBenchmark(autoload=False, subset=[f"matbench_{task}"])
    t = list(mb.tasks)[0]
    t.load()
    df = t.df
    target = t.metadata.target
    inputs = df[df.columns[0]].tolist()
    ids = [str(i) for i in df.index]
    y = df[target].astype(float).to_numpy()
    folds_ids = json.load(open(os.path.join(META, "folds.json")))[f"matbench_{task}"]
    pos = {m: i for i, m in enumerate(ids)}
    folds = [np.array(sorted(pos[m] for m in f)) for f in folds_ids]
    return ids, inputs, y, folds


def summary(task: str) -> None:
    d = os.path.join(OUT, task)
    files = sorted(f for f in os.listdir(d) if f.startswith("fold") and f[4:].split(".")[0].isdigit()) \
        if os.path.isdir(d) else []
    res = [json.load(open(os.path.join(d, f))) for f in files]
    if not res:
        print(f"{task}: no results yet")
        return
    print(f"\n{task} ({UNITS[task]}), {len(res)} fold(s)")
    for k in res[0]["test"]:
        v = [r["test"][k] for r in res]
        print(f"  QFA {k:22s} {np.mean(v):9.4f}  {[round(x, 4) for x in v]}")
    for name, v in sorted(leaderboard_fold_scores(task).items(), key=lambda kv: np.mean(kv[1])):
        print(f"  LB  {name:22s} {np.mean(v[:len(res)]):9.4f}  (same folds)")


def export(tasks, path: str) -> None:
    """One JSON with every finished result, for the results slides: test scores per variant and fold,
    leaderboard scores on the same folds, trained numbers, gates and CPU time."""
    out = {}
    for task in tasks:
        d = os.path.join(OUT, task)
        if not os.path.isdir(d):
            continue
        rows = {}
        for f in sorted(os.listdir(d)):
            if f.endswith(".json") and f.startswith("fold") and "quick" not in f:
                rows[f[:-5]] = json.load(open(os.path.join(d, f)))
        if not rows:
            continue
        t = {"runs": {}, "leaderboard": leaderboard_fold_scores(task), "unit": UNITS[task]}
        for name, r in rows.items():
            seeds = r["per_seed"]
            t["runs"][name] = {
                "test": r["test"], "seconds": r["seconds"], "tier": r["tier"], "ablation": r.get("ablation", "none"),
                "params": seeds[0]["params"], "params_pruned": seeds[0].get("params_pruned"),
                "gates_dense": seeds[0].get("gates_dense"), "gates_pruned": seeds[0].get("gates_pruned"),
                "qt_trained": seeds[0].get("qt_trained"), "qt_qubits": seeds[0].get("qt_qubits"),
                "qt_materialised_equal": [s.get("qt_materialised_equal") for s in seeds],
                "depth_weights": [s.get("depth_weights") for s in seeds],
                "sec_by_phase": {k: float(np.mean([s.get(k, 0) for s in seeds]))
                                 for k in ("sec_classical", "sec_hybrid_and_pruning", "sec_qt")},
            }
        cache = os.path.join(os.path.dirname(OUT), "..", "data", "cache")
        for fn in os.listdir(cache) if os.path.isdir(cache) else []:
            if fn.startswith(f"qfa_{task}_") and fn.endswith("_time.json"):
                t["backbone_time"] = json.load(open(os.path.join(cache, fn)))
        out[task] = t
    json.dump(out, open(path, "w"), indent=1, default=float)
    print(f"wrote {path} ({len(out)} tasks)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="a MatBench task name without 'matbench_', or 'all'")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=None, help="cap per training stage")
    ap.add_argument("--backbone", default="orb_v2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ablation", default="none", choices=["none", "last_layer", "no_adapter", "no_desc"],
                    help="classical ablations (the quantum controls run in every job)")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--export", default=None, help="write all finished results to this JSON file")
    a = ap.parse_args()
    tasks = TASKS if a.task == "all" else [a.task]
    if a.summary or a.export:
        for t in tasks:
            summary(t)
        if a.export:
            export(tasks, a.export)
        return
    from src.qfa.data import prepare_task
    from src.qfa.train import run_fold
    for task in tasks:
        ids, inputs, y, folds = load_matbench(task)
        prep = prepare_task(task, ids, inputs, backbone=a.backbone)
        os.makedirs(os.path.join(OUT, task), exist_ok=True)
        for f in [int(x) for x in a.folds.split(",")]:
            tag = ("" if a.ablation == "none" else f"_{a.ablation}") + ("_quick" if a.quick else "")
            path = os.path.join(OUT, task, f"fold{f}{tag}.json")
            if os.path.exists(path):
                print(f"{task} fold {f}: done")
                continue
            print(f"{task} fold {f}", flush=True)
            info = run_fold(task, prep, prep["desc"], y, folds[f], seeds=a.seeds, quick=a.quick,
                            max_minutes=a.max_minutes, device=a.device, log=lambda m: print(m, flush=True),
                            ablation=a.ablation)
            json.dump(info, open(path, "w"), indent=1, default=float)
            print(f"{task} fold {f}: {info['test']}", flush=True)
        summary(task)


if __name__ == "__main__":
    main()
