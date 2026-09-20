"""Feature archive discovery and loading utilities.

The ordering in this module is part of the legacy experiment contract. In
particular, callers rely on both the candidate-path order and the NPZ key
fallback order when loading previously generated feature archives.
"""

from __future__ import annotations

import os

import numpy as np

DEFAULT_DATA_ROOT = os.environ.get(
    "MIXTURE_GREEDY_DATA_ROOT",
    "/research/d7/rshr/bahar/data/dgm",
)


def candidate_feature_npz_paths(dataset_name, model_name, feature_extractor):
    """Return feature archive candidates in legacy lookup order."""
    base = os.path.join(DEFAULT_DATA_ROOT, dataset_name, "features")
    candidates = [
        os.path.join(base, feature_extractor, f"{model_name}.npz"),
        os.path.join(base, feature_extractor.lower(), f"{model_name}.npz"),
        os.path.join(base, "Inception", f"{model_name}.npz"),
        os.path.join(base, "inception", f"{model_name}.npz"),
        os.path.join(base, f"{model_name}.npz"),
    ]

    # Do not replace this with a set: insertion order and duplicate removal are
    # observable behavior used by the legacy loaders.
    unique_candidates: list[str] = []
    for path in candidates:
        if path not in unique_candidates:
            unique_candidates.append(path)
    return unique_candidates


def resolve_existing_path(candidates):
    """Return the first existing candidate, or the first candidate as fallback."""
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def load_features_from_npz(npz_path, feature_extractor, return_key=False):
    """Load a feature array using the legacy key-precedence contract.

    If none of the recognized keys exists, the first array in the archive is
    returned. An empty archive raises ``KeyError`` exactly as the legacy
    implementation did.
    """
    data = np.load(npz_path)
    preferred_key = f"{feature_extractor}_features"
    fallback_keys = [
        preferred_key,
        f"{feature_extractor.lower()}_features",
        "dino_features",
        "inception_features",
        "clip_features",
        "sbert_features",
        "features",
    ]
    for key in fallback_keys:
        if key in data:
            if return_key:
                return data[key], key
            return data[key]

    if len(data.files) > 0:
        first_key = data.files[0]
        if return_key:
            return data[first_key], first_key
        return data[first_key]

    raise KeyError(f"No feature arrays found in {npz_path}")


# Private aliases retained so legacy modules can re-export their original API.
_candidate_feature_npz_paths = candidate_feature_npz_paths
_resolve_existing_path = resolve_existing_path
_load_features_from_npz = load_features_from_npz

__all__ = [
    "DEFAULT_DATA_ROOT",
    "candidate_feature_npz_paths",
    "load_features_from_npz",
    "resolve_existing_path",
]
