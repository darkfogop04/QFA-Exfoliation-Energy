#!/usr/bin/env python
"""
Cost-reduction pipeline for the HQ-GNN on one MatBench fold:

  1. dense   - train the hybrid model (validation split, test fold untouched)
  2. prune   - iteratively prune the VQC, fine-tuning after each round
  3. qt      - retrain from scratch with Quantum-Train generating the classical
               weights, keeping the pruned circuit structure; then materialise
               the weights and evaluate the purely classical-weight model
  (+ optional QT controls: generator=classical / none)

Every stage reports test MAE/RMSE/R2 and the cost columns the thesis needs:
trainable params, quantum params, 1q/2q gates, depth, s/epoch, ms/crystal.

Usage
  python experiments/run_optimization.py --config configs/optimize.yaml --fold 0
  python experiments/run_optimization.py --config configs/optimize.yaml --fold 0 \
         --stages dense,prune,qt --qt-controls --seed 1
"""
import argparse
import copy
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.data.dataset import MatBenchExfoliationDataset
from src.models.hybrid_model import HybridQuantumGNN
from src.optimization.pruning import iterative_prune
from src.optimization.quantum_train import QuantumTrainModel
from src.optimization.size_policy import resolve_profile
from src.training.trainer import Trainer
from src.utils.config import get_config
from src.utils.seed import set_seed

logger = logging.getLogger("run_optimization")


def count_trainable(model) -> int:
    """Trainable parameters, counting only circuit angles that are still in the
    circuit (pruned and dead angles stay in the weight tensor but do nothing)."""
    total, q_layer = 0, None
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("quantum.weights") or name.endswith("quantum.ent_weights"):
            continue
        total += p.numel()
    for m in model.modules():
        if hasattr(m, "resource_counts") and hasattr(m, "active_mask"):
            q_layer = m
    if q_layer is not None:
        total += q_layer.resource_counts()["quantum_params"]
    return total


@torch.no_grad()
def ms_per_crystal(model, loader, device="cpu") -> float:
    model.eval()
    n, t0 = 0, time.perf_counter()
    for batch in loader:
        batch = batch.to(device)
        model(batch)
        n += batch.num_graphs
    return 1000 * (time.perf_counter() - t0) / max(1, n)


def train_and_test(model, cfg, loaders, stats, device):
    train_loader, val_loader, test_loader = loaders
    trainer = Trainer(model, cfg, device=device)
    t0 = time.perf_counter()
    res = trainer.train(train_loader, val_loader, *stats)
    dt = time.perf_counter() - t0
    test = trainer.evaluate(test_loader, *stats)
    epochs = max(1, len(res["train_history"]))
    return {
        "val_mae": (res.get("best_metrics") or {}).get("mae"),
        **{f"test_{k}": v for k, v in test.items()},
        "epochs": epochs,
        "s_per_epoch": dt / epochs,
        "train_minutes": dt / 60,
        "stopped_by_time": getattr(trainer, "stopped_by_time", False),
    }


def row(stage, model, quantum_layer, metrics, test_loader, device, extra=None):
    r = {"stage": stage, **metrics, **quantum_layer.resource_counts(),
         "trainable_total": count_trainable(model),
         "ms_per_crystal": ms_per_crystal(model, test_loader, device)}
    r.update(extra or {})
    logger.info(json.dumps(r, default=str))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/optimize.yaml")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--stages", default="dense,prune,qt")
    ap.add_argument("--qt-controls", action="store_true",
                    help="also run QT with generator=classical and generator=none")
    ap.add_argument("--out", default="results")
    ap.add_argument("--task", default=None, help="MatBench task, e.g. matbench_phonons (overrides data.task_name)")
    ap.add_argument("--variant", default="pipeline", choices=["pipeline", "midterm"],
                    help="pipeline: new circuit -> pruning -> QT; midterm: the mid-term model (fair test), dense only")
    ap.add_argument("--adaptive", action="store_true", help="apply the size policy (src/optimization/size_policy.py)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override training.epochs (quick trials)")
    ap.add_argument("--max-minutes", type=float, default=None, help="time budget per training run (dense, each fine-tune)")
    ap.add_argument("--qt-max-minutes", type=float, default=None, help="time budget for each Quantum-Train run")
    ap.add_argument("--max-train", type=int, default=None, help="cap training crystals (quick trials only)")
    ap.add_argument("--max-eval", type=int, default=None, help="cap validation/test crystals (quick trials only)")
    ap.add_argument("--finetune-epochs", type=int, default=None, help="override pruning.finetune_epochs")
    ap.add_argument("--force-qt", action="store_true", help="run Quantum-Train even if the size policy turns it off")
    ap.add_argument("--ckpt-in", default=None, help="load this model state instead of training the dense model")
    ap.add_argument("--ckpt-out", default=None, help="save the final (dense or pruned) model state here")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cfg = get_config(args.config)
    if args.seed is not None:
        cfg.setdefault("training", {})["seed"] = args.seed
    seed = cfg.get("training", {}).get("seed", 42)
    stages = set(args.stages.split(","))
    device = "cpu"

    d = cfg.setdefault("data", {})
    t = cfg.setdefault("training", {})
    if args.task:
        d["task_name"] = args.task
    if args.batch_size:
        d["batch_size"] = args.batch_size
    if args.max_train:
        d["max_train"] = args.max_train
    if args.max_eval:
        d["max_eval"] = args.max_eval
    if args.epochs:
        t["epochs"] = args.epochs
        cfg.setdefault("qt", {}).setdefault("training", {})["epochs"] = args.epochs
    if args.max_minutes:
        t["max_minutes"] = args.max_minutes
    if args.qt_max_minutes:
        cfg.setdefault("qt", {}).setdefault("training", {})["max_minutes"] = args.qt_max_minutes
    if args.variant == "midterm":
        # the mid-term model: raw angle encoding, no L1, 3 layers, no pruning / QT, no size policy
        cfg["quantum"].update({"input_scaling": "none", "l1_angles": 0.0, "n_layers": 3})
        cfg.setdefault("adaptive", {})["enabled"] = False
        stages = {"dense"}
    elif args.adaptive:
        cfg.setdefault("adaptive", {})["enabled"] = True
    if args.finetune_epochs:
        cfg.setdefault("pruning", {})["finetune_epochs_override"] = args.finetune_epochs
    task_name = d.get("task_name", "matbench_jdft2d")
    tag = f"{task_name}_fold{args.fold}_seed{seed}_{args.variant}"

    dataset = MatBenchExfoliationDataset(cfg)
    loaders = dataset.load_fold_with_val(args.fold, cfg.get("data", {}).get("val_fraction", 0.1), seed)
    stats = (dataset.target_mean, dataset.target_std)
    n_train = len(loaders[0].dataset)

    if cfg.get("adaptive", {}).get("enabled", False):
        cfg = resolve_profile(n_train, cfg)
        logger.info(f"Size profile: {cfg['adaptive']['resolved_tier']} (n_train={n_train})")
    if args.force_qt:
        cfg.setdefault("qt", {})["enabled"] = True
    if cfg.get("pruning", {}).get("finetune_epochs_override"):
        cfg["pruning"]["finetune_epochs"] = cfg["pruning"]["finetune_epochs_override"]

    rows = []
    info = {"task": task_name, "fold": args.fold, "seed": seed, "variant": args.variant,
            "n_train": n_train, "n_val": len(loaders[1].dataset), "n_test": len(loaders[2].dataset),
            "tier": cfg.get("adaptive", {}).get("resolved_tier")}

    # 1. dense --------------------------------------------------------
    set_seed(seed)
    dense = HybridQuantumGNN(cfg).to(device)
    if args.ckpt_in:
        state = torch.load(args.ckpt_in, map_location=device, weights_only=False)
        dense.load_state_dict(state)
        test = Trainer(dense, cfg, device=device).evaluate(loaders[2], *stats)
        logger.info(f"Loaded {args.ckpt_in}: test {test}")
        rows.append(row("loaded", dense, dense.quantum, {f"test_{k}": v for k, v in test.items()}, loaders[2], device))
    elif "dense" in stages:
        m = train_and_test(dense, cfg, loaders, stats, device)
        rows.append(row("dense", dense, dense.quantum, m, loaders[2], device))

    # 2. prune --------------------------------------------------------
    if "prune" in stages and cfg.get("pruning", {}).get("enabled", True):
        history = iterative_prune(dense, cfg, loaders[0], loaders[1], *stats, device=device)
        test = Trainer(dense, cfg, device=device).evaluate(loaders[2], *stats)
        rows.append(row("pruned", dense, dense.quantum,
                        {f"test_{k}": v for k, v in test.items()}, loaders[2], device,
                        {"prune_rounds": len(history) - 1}))
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"prune_history_{tag}.json"), "w") as f:
            json.dump(history, f, indent=2, default=str)

    # 3. Quantum-Train ------------------------------------------------
    if "qt" in stages and cfg.get("qt", {}).get("enabled", True):
        generators = ["quantum"] + (["classical", "none"] if args.qt_controls else [])
        for gen in generators:
            qt_cfg = copy.deepcopy(cfg.get("qt", {}))
            qt_cfg["generator"] = gen
            set_seed(seed)
            base = HybridQuantumGNN(cfg).to(device)
            # keep the pruned circuit structure; angles are retrained from scratch
            base.quantum.set_mask(dense.quantum.mask,
                                  dense.quantum.ent_mask if base.quantum.entangler == "crx" else None)
            qt_model = QuantumTrainModel(base, qt_cfg)
            train_cfg = copy.deepcopy(cfg)
            train_cfg["training"].update(qt_cfg.get("training", {}))
            m = train_and_test(qt_model, train_cfg, loaders, stats, device)
            breakdown = qt_model.trainable_breakdown()
            rows.append(row(f"qt_{gen}", qt_model, base.quantum, m, loaders[2], device, breakdown))

            # materialised model: QNN dropped, must give the same predictions
            final = qt_model.materialize()
            test = Trainer(final, cfg, device=device).evaluate(loaders[2], *stats)
            rows.append(row(f"qt_{gen}_materialized", final, final.quantum,
                            {f"test_{k}": v for k, v in test.items()}, loaders[2], device,
                            {"trainable_total": breakdown["total_trainable"]}))

    os.makedirs(args.out, exist_ok=True)
    if args.ckpt_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.ckpt_out)), exist_ok=True)
        torch.save(dense.state_dict(), args.ckpt_out)
        logger.info(f"Saved model state to {args.ckpt_out}")
    path = os.path.join(args.out, f"optimization_{tag}.json")
    with open(path, "w") as f:
        json.dump({"info": info, "config": cfg, "rows": rows}, f, indent=2, default=str)
    logger.info(f"Saved {path}")

    # compact table
    cols = ["stage", "test_mae", "test_r2", "trainable_total", "quantum_params",
            "two_qubit_gates", "total_gates", "depth", "ms_per_crystal"]
    print("\n" + " | ".join(cols))
    for r in rows:
        print(" | ".join(f"{r.get(c):.4g}" if isinstance(r.get(c), float) else str(r.get(c)) for c in cols))


if __name__ == "__main__":
    main()
