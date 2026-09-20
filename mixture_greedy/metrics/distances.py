"""Shared distance calculations for model-quality metrics."""

from __future__ import annotations

import sklearn.metrics


def compute_pairwise_distance(data_x, data_y=None):
    """Compute the Euclidean pairwise-distance matrix used by legacy metrics."""
    if data_y is None:
        data_y = data_x
    return sklearn.metrics.pairwise_distances(
        data_x,
        data_y,
        metric="euclidean",
        n_jobs=8,
    )


__all__ = ["compute_pairwise_distance"]
