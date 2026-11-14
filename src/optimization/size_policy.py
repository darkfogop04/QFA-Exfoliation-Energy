"""
Size-adaptive profile: choose circuit depth, pruning aggressiveness and
whether to use Quantum-Train from the number of training samples.

Reasoning behind the default tiers
  * Small N: the risk is over-fitting, so capacity must be *removed*.
    Shallow circuit, aggressive pruning, and Quantum-Train on the whole
    classical network (73k weights -> ~2k trainable).
  * Large N: over-fitting matters less, but the VQC is simulated once per
    crystal per step, so its cost grows with N. Keep more capacity
    (deeper circuit), use pruning mainly to cut gates/time, and skip QT
    (its regularisation is no longer needed and its per-step 17-qubit
    simulation multiplies over ~8k steps per epoch on mp_e_form).

These tiers are a starting hypothesis for Phase 3, not a result. Override
them in the YAML under `adaptive.tiers` and fit the rule from the sweep.
"""
import copy
from typing import Any, Dict, List

from src.utils.config import _update_dict

DEFAULT_TIERS: List[Dict[str, Any]] = [
    {   # jdft2d (636 -> ~458 train after fold + validation split)
        "name": "tiny", "max_train": 800,
        "overrides": {
            "quantum": {"n_layers": 2},
            "pruning": {"enabled": True, "max_sparsity": 0.6, "tolerance": 0.03},
            "qt": {"enabled": True, "targets": ["gnn", "mlp"], "n_qnn_layers": 8},
            "mlp": {"dropout": 0.2},
        },
    },
    {   # phonons (~911 train), dielectric (~3.4k train)
        "name": "small", "max_train": 5_000,
        "overrides": {
            "quantum": {"n_layers": 3},
            "pruning": {"enabled": True, "max_sparsity": 0.5, "tolerance": 0.02},
            "qt": {"enabled": True, "targets": ["gnn.convs"], "n_qnn_layers": 16},
        },
    },
    {   # log_gvrh (11k), perovskites (19k)
        "name": "medium", "max_train": 25_000,
        "overrides": {
            "quantum": {"n_layers": 3},
            "pruning": {"enabled": True, "max_sparsity": 0.4, "tolerance": 0.01},
            "qt": {"enabled": False},
        },
    },
    {   # mp_e_form (133k)
        "name": "large", "max_train": float("inf"),
        "overrides": {
            "quantum": {"n_layers": 4},
            "pruning": {"enabled": True, "max_sparsity": 0.3, "tolerance": 0.01},
            "qt": {"enabled": False},
        },
    },
]


def resolve_profile(n_train: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of `config` with the tier for `n_train` merged in.

    Keys the user set explicitly under `quantum`, `pruning`, `qt` in the YAML
    are overridden by the tier; put per-task exceptions in `adaptive.tiers`.
    """
    cfg = copy.deepcopy(config)
    tiers = cfg.get("adaptive", {}).get("tiers") or DEFAULT_TIERS
    tier = next(t for t in tiers if n_train <= t["max_train"])
    _update_dict(cfg, copy.deepcopy(tier["overrides"]))
    cfg.setdefault("adaptive", {})["resolved_tier"] = tier["name"]
    cfg["adaptive"]["n_train"] = n_train
    return cfg
