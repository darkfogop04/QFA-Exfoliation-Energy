"""
Iterative pruning of the VQC (rotation angles and, with entangler="crx",
the two-qubit gates).

Loop (a simplified, regression-friendly version of PECT, Sim et al. 2021):

    train dense model (validation split, NOT the test fold)
    repeat:
        score every active angle            (ablation / magnitude / taylor)
        remove the lowest-scoring fraction  (gates disappear from the circuit)
        optionally regrow a few at random   (PECT-style exploration, start at 0)
        fine-tune the whole network for a few epochs
        if val MAE got worse than (1 + tolerance) * dense val MAE: undo, stop
    keep the smallest circuit that stayed within tolerance

Scoring methods
  ablation  - remove the gate, measure the increase in validation loss.
              Exact and cheap here: the GNN does not depend on the angles, so
              latents are computed once and only the circuit + MLP head rerun.
  magnitude - |angle| after wrapping to (-pi, pi] (rotations are 2*pi-periodic
              up to a global phase, so 6.2 rad is "small", not "large").
  taylor    - |theta * dL/dtheta|, first-order estimate of the loss change
              when theta -> 0. One backward pass.
"""
import copy
import logging
import math
import time
from typing import Any, Dict, List, Optional

import torch

from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


def wrap_angle(theta: torch.Tensor, period: float = 2 * math.pi) -> torch.Tensor:
    """Map angles into (-period/2, period/2]."""
    half = period / 2
    return torch.remainder(theta + half, period) - half


# --------------------------------------------------------------- latents
@torch.no_grad()
def collect_latents(model, loader, device: str = "cpu"):
    """Run the GNN once; return (latents, standardized targets)."""
    model.eval()
    xs, ys = [], []
    for batch in loader:
        batch = batch.to(device)
        xs.append(model.encode(batch).detach())
        ys.append(batch.y.float().view(-1).cpu())
    return torch.cat(xs), torch.cat(ys)


@torch.no_grad()
def head_loss(model, X: torch.Tensor, y: torch.Tensor, chunk: int = 256) -> float:
    model.eval()
    preds = torch.cat([model.head(X[i:i + chunk]).cpu() for i in range(0, len(X), chunk)])
    return torch.mean((preds - y) ** 2).item()


# ---------------------------------------------------------------- scores
def score_angles(model, X: torch.Tensor, y: torch.Tensor, method: str = "ablation") -> Dict[str, Optional[torch.Tensor]]:
    """Importance of every active angle. Inactive entries get +inf."""
    layer = model.quantum
    active = layer.active_mask()
    rot = torch.full(layer.weights.shape, float("inf"))
    ent = None
    if layer.entangler == "crx":
        ent = torch.full(layer.ent_mask.shape, float("inf"))

    if method == "magnitude":
        mag = wrap_angle(layer.weights.detach().cpu()).abs()
        rot[active.cpu()] = mag[active.cpu()]
        if ent is not None:
            # CRX is 4*pi-periodic; CRX(0) is the identity.
            emag = wrap_angle(layer.ent_weights.detach().cpu(), 4 * math.pi).abs()
            ent[layer.ent_mask.cpu()] = emag[layer.ent_mask.cpu()]

    elif method == "taylor":
        model.eval()
        layer.weights.grad = None
        if layer.ent_weights is not None:
            layer.ent_weights.grad = None
        preds = model.head(X)
        loss = torch.mean((preds.cpu() - y) ** 2)
        loss.backward()
        t = (wrap_angle(layer.weights.detach()) * layer.weights.grad).abs().cpu()
        rot[active.cpu()] = t[active.cpu()]
        if ent is not None:
            te = (wrap_angle(layer.ent_weights.detach(), 4 * math.pi) * layer.ent_weights.grad).abs().cpu()
            ent[layer.ent_mask.cpu()] = te[layer.ent_mask.cpu()]
        model.zero_grad(set_to_none=True)

    elif method == "ablation":
        base = head_loss(model, X, y)
        for idx in active.nonzero().tolist():
            l, i, j = idx
            layer.mask[l, i, j] = False          # gate removed, weight kept
            rot[l, i, j] = head_loss(model, X, y) - base
            layer.mask[l, i, j] = True
        if ent is not None:
            for l, i in layer.ent_mask.nonzero().tolist():
                layer.ent_mask[l, i] = False
                ent[l, i] = head_loss(model, X, y) - base
                layer.ent_mask[l, i] = True
    else:
        raise ValueError(f"Unknown scoring method {method!r}")

    return {"rot": rot, "ent": ent}


def prune_lowest(layer, scores: Dict[str, Optional[torch.Tensor]], n_prune: int,
                 regrow: int = 0, generator: Optional[torch.Generator] = None) -> int:
    """Remove the n_prune lowest-scoring gates (rotations and entanglers ranked
    together), then re-activate `regrow` random pruned rotations at angle 0."""
    rot, ent = scores["rot"], scores["ent"]
    flat = [rot.flatten()]
    if ent is not None:
        flat.append(ent.flatten())
    allscores = torch.cat(flat)
    finite = torch.isfinite(allscores)
    n_prune = min(n_prune, int(finite.sum()))
    if n_prune <= 0:
        return 0
    order = torch.argsort(torch.where(finite, allscores, torch.tensor(float("inf"))))
    chosen = order[:n_prune]

    new_mask = layer.mask.clone().cpu()
    new_ent = layer.ent_mask.clone().cpu() if ent is not None else None
    n_rot = rot.numel()
    for k in chosen.tolist():
        if k < n_rot:
            new_mask.view(-1)[k] = False
        else:
            new_ent.view(-1)[k - n_rot] = False

    if regrow > 0:
        dead = layer.dead_mask().cpu()
        just_pruned = set(k for k in chosen.tolist() if k < n_rot)
        candidates = [k for k in (~new_mask & ~dead).view(-1).nonzero().flatten().tolist()
                      if k not in just_pruned]
        if candidates:
            perm = torch.randperm(len(candidates), generator=generator)[:regrow]
            for p in perm.tolist():
                new_mask.view(-1)[candidates[p]] = True  # set_mask leaves it at 0

    layer.set_mask(new_mask, new_ent)
    return n_prune


# ------------------------------------------------------------- main loop
def _record(round_idx: int, layer, metrics: Dict[str, float], extra: Dict[str, Any]) -> Dict[str, Any]:
    row = {"round": round_idx}
    row.update(layer.resource_counts())
    row.update({f"val_{k}": v for k, v in metrics.items()})
    row.update(extra)
    return row


def iterative_prune(model, config: Dict[str, Any], train_loader, val_loader,
                    target_mean: float, target_std: float, device: str = "cpu") -> List[Dict[str, Any]]:
    """Prune `model.quantum` in place. Returns one row per round (round 0 = dense)."""
    pcfg = config.get("pruning", {})
    method = pcfg.get("method", "ablation")
    frac = pcfg.get("fraction_per_round", 0.2)
    max_sparsity = pcfg.get("max_sparsity", 0.6)
    tol = pcfg.get("tolerance", 0.02)
    ft_epochs = pcfg.get("finetune_epochs", 30)
    regrow_frac = pcfg.get("regrow_fraction", 0.0)
    gen = torch.Generator().manual_seed(config.get("training", {}).get("seed", 42))

    layer = model.quantum
    n_start = layer.resource_counts()["quantum_params"]
    n_floor = max(1, math.ceil((1 - max_sparsity) * n_start))

    ft_config = copy.deepcopy(config)
    ft_config.setdefault("training", {})["epochs"] = ft_epochs
    ft_config["training"]["patience"] = min(ft_config["training"].get("patience", 20), ft_epochs)
    if "finetune_lr" in pcfg:
        ft_config["training"]["lr"] = pcfg["finetune_lr"]

    base_metrics = Trainer(model, config, device=device).evaluate(val_loader, target_mean, target_std)
    base_mae = base_metrics["mae"]
    history = [_record(0, layer, base_metrics, {"finetune_s": 0.0})]
    best_state = copy.deepcopy(model.state_dict())
    logger.info(f"[prune] dense: {history[0]}")

    r = 0
    while True:
        active = layer.resource_counts()["quantum_params"]
        if active <= n_floor:
            break
        r += 1
        n_prune = min(max(1, round(frac * active)), active - n_floor)
        n_regrow = int(round(regrow_frac * n_prune))

        X, y = collect_latents(model, val_loader, device)
        scores = score_angles(model, X, y, method=method)
        prune_lowest(layer, scores, n_prune + n_regrow, regrow=n_regrow, generator=gen)

        t0 = time.perf_counter()
        trainer = Trainer(model, ft_config, device=device)
        trainer.train(train_loader, val_loader, target_mean, target_std)
        dt = time.perf_counter() - t0
        metrics = trainer.evaluate(val_loader, target_mean, target_std)
        row = _record(r, layer, metrics, {"finetune_s": dt})
        history.append(row)
        logger.info(f"[prune] round {r}: {row}")

        if metrics["mae"] > (1 + tol) * base_mae:
            logger.info(f"[prune] val MAE {metrics['mae']:.3f} > {(1 + tol) * base_mae:.3f}; undoing round {r}")
            row["accepted"] = False
            break
        row["accepted"] = True
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return history
