"""Unified entry point for metric-specific online evaluations.

The evaluator owns sampling and optimization. Modules under ``metrics`` only
implement numerical objectives used by those evaluators.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mixture_greedy.config import Config, ConfigFID, ConfigVNE

SUPPORTED_METRICS = ("rke", "kid", "fid", "vne")
SUPPORTED_MODES = {
    "rke": (
        "mixture-greedy",
        "mixture-greedy-eg",
        "mixture-ucb",
        "mixture-oracle",
        "one-arm-ucb",
        "one-arm-oracle",
    ),
    "kid": (
        "mixture-greedy",
        "mixture-greedy-eg",
        "mixture-ucb",
        "mixture-oracle",
        "one-arm-ucb",
        "one-arm-oracle",
    ),
    "fid": (
        "mixture-greedy",
        "mixture-oracle",
        "one-arm-greedy",
        "one-arm-eps-greedy",
        "one-arm-epsilon-greedy",
    ),
    "vne": (
        "mixture-greedy",
        "mixture-oracle",
        "one-arm-greedy",
        "one-arm-eps-greedy",
        "one-arm-epsilon-greedy",
    ),
}


@dataclass(frozen=True)
class EvaluationResult:
    """Common result returned by every supported evaluator."""

    metric: str
    scores: np.ndarray
    alpha_history: np.ndarray
    sample_sizes: np.ndarray
    evaluator: Any

    def save(self, output_path: str | Path) -> None:
        """Save metric-independent result keys to one NPZ archive."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            metric=self.metric,
            scores=self.scores,
            alpha_history=self.alpha_history,
            sample_sizes=self.sample_sizes,
        )


def create_config(metric: str):
    """Create the configuration class associated with ``metric``."""
    metric = _normalize_metric(metric)
    if metric in {"rke", "kid"}:
        return Config(QUADRATIC_METRIC=metric)
    if metric == "fid":
        return ConfigFID()
    return ConfigVNE()


def create_evaluator(
    metric: str,
    model_names: Sequence[str],
    dataset_name: str,
    *,
    mode: str = "mixture-greedy",
    config=None,
    offline_evaluator=None,
    real_dataset_path: str | None = None,
    oracle_alphas=None,
):
    """Build the correct evaluator while keeping metric-specific setup local."""
    metric = _normalize_metric(metric)
    mode = _normalize_mode(metric, mode)
    config = config or create_config(metric)

    if metric in {"rke", "kid"}:
        from mixture_greedy.rke_offline import RKEOfflineEvaluator
        from mixture_greedy.rke_online import RKEOnlineEvaluator

        config.QUADRATIC_METRIC = metric
        use_linear = metric == "kid"
        if offline_evaluator is None:
            offline_evaluator = RKEOfflineEvaluator(
                model_names,
                dataset_name,
                has_reference=use_linear,
                config=config,
            )
        return RKEOnlineEvaluator(
            model_names,
            dataset_name,
            offline_evaluator,
            mode,
            use_linear=use_linear,
            config=config,
        )

    if metric == "fid":
        from FID_online import FIDOnlineEvaluator, SimpleOfflineEvaluator

        if offline_evaluator is None:
            if real_dataset_path is None:
                raise ValueError("FID evaluation requires real_dataset_path")
            offline_evaluator = SimpleOfflineEvaluator(
                real_dataset_path,
                feature_key=f"{config.feature_extractor}_features",
                optimal_alphas=oracle_alphas,
            )
        return FIDOnlineEvaluator(
            model_names,
            dataset_name,
            offline_evaluator,
            mode=mode,
            config=config,
            optimal_alphas=oracle_alphas,
        )

    from VNE_online_new import VNEOnlineEvaluator

    return VNEOnlineEvaluator(
        model_names,
        dataset_name,
        offline_evaluator=offline_evaluator,
        mode=mode,
        config=config,
        oracle_alphas=oracle_alphas,
    )


def run_evaluation(
    metric: str,
    model_names: Sequence[str],
    dataset_name: str,
    *,
    rounds: int | None = None,
    mode: str = "mixture-greedy",
    config=None,
    offline_evaluator=None,
    real_dataset_path: str | None = None,
    oracle_alphas=None,
) -> EvaluationResult:
    """Create and run one RKE, KID, FID, or VNE evaluation."""
    normalized_metric = _normalize_metric(metric)
    config = config or create_config(normalized_metric)
    evaluator = create_evaluator(
        normalized_metric,
        model_names,
        dataset_name,
        mode=mode,
        config=config,
        offline_evaluator=offline_evaluator,
        real_dataset_path=real_dataset_path,
        oracle_alphas=oracle_alphas,
    )
    evaluator.run_online_evaluation(
        num_rounds=config.TOTAL_ROUNDS if rounds is None else rounds
    )

    score_attribute = {
        "rke": "scores",
        "kid": "scores",
        "fid": "fid_scores",
        "vne": "vne_scores",
    }[normalized_metric]
    return EvaluationResult(
        metric=normalized_metric,
        scores=np.asarray(getattr(evaluator, score_attribute)),
        alpha_history=np.asarray(evaluator.alpha_history),
        sample_sizes=np.asarray(evaluator._get_sample_sizes()),
        evaluator=evaluator,
    )


def _normalize_metric(metric: str) -> str:
    normalized = metric.lower().strip()
    if normalized not in SUPPORTED_METRICS:
        choices = ", ".join(SUPPORTED_METRICS)
        raise ValueError(f"Unsupported metric {metric!r}; choose one of: {choices}")
    return normalized


def _normalize_mode(metric: str, mode: str) -> str:
    normalized = mode.lower().strip()
    if normalized not in SUPPORTED_MODES[metric]:
        choices = ", ".join(SUPPORTED_MODES[metric])
        raise ValueError(
            f"Unsupported mode {mode!r} for {metric}; choose one of: {choices}"
        )
    return normalized


__all__ = [
    "EvaluationResult",
    "SUPPORTED_METRICS",
    "SUPPORTED_MODES",
    "create_config",
    "create_evaluator",
    "run_evaluation",
]
