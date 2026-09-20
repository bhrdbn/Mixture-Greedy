"""Online mixture selection for generative-model evaluation.

The package offers a stable import surface while the original top-level
research scripts remain available for backward compatibility. Heavy evaluator
modules are imported lazily so importing data utilities does not initialize
Torch, CVXPY, SciPy, or Matplotlib.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .data import (
    candidate_feature_npz_paths,
    load_features_from_npz,
    resolve_existing_path,
)

__version__ = "0.1.0"

_LAZY_EXPORTS = {
    "Config": ("mixture_greedy.config", "Config"),
    "KernelUtils": ("mixture_greedy.metrics.rke", "KernelUtils"),
    "RKEOfflineEvaluator": (
        "mixture_greedy.rke_offline",
        "RKEOfflineEvaluator",
    ),
    "RKEOnlineEvaluator": (
        "mixture_greedy.rke_online",
        "RKEOnlineEvaluator",
    ),
    "ConfigFID": ("mixture_greedy.config", "ConfigFID"),
    "FIDOnlineEvaluator": ("FID_online", "FIDOnlineEvaluator"),
    "ConfigVNE": ("mixture_greedy.config", "ConfigVNE"),
    "VNEOnlineEvaluator": ("VNE_online_new", "VNEOnlineEvaluator"),
    "EvaluationResult": ("mixture_greedy.runner", "EvaluationResult"),
    "create_config": ("mixture_greedy.runner", "create_config"),
    "create_evaluator": ("mixture_greedy.runner", "create_evaluator"),
    "run_evaluation": ("mixture_greedy.runner", "run_evaluation"),
}


def __getattr__(name: str) -> Any:
    """Load evaluator APIs on first access."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "Config",
    "ConfigFID",
    "ConfigVNE",
    "EvaluationResult",
    "FIDOnlineEvaluator",
    "KernelUtils",
    "RKEOfflineEvaluator",
    "RKEOnlineEvaluator",
    "VNEOnlineEvaluator",
    "candidate_feature_npz_paths",
    "create_config",
    "create_evaluator",
    "load_features_from_npz",
    "resolve_existing_path",
    "run_evaluation",
]
