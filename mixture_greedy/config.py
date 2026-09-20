"""Configuration objects for the supported experiment families.

The three classes intentionally remain separate: similarly named settings have
different defaults and meanings in the KID/RKE, FID, and VNE experiments.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class Config:
    """Configuration for RKE evaluation."""

    OGD: bool = False
    ALPHA_PRINT: int = 1000
    RESCALER: float = 1
    KID_NEW: bool = False
    QUADRATIC_METRIC: str = "rke"
    LINEAR_METRIC: str = "precision"
    NUMBER_OF_SIMULATIONS: int = 1
    SIGMA: float = 5
    DELTA: float = 0.03
    BETA: float = 1
    INITIAL_SAMPLE_COUNT: int = 3
    OFFLINE_CUTOFF: int = 10000
    DATASET_REAL_CUTOFF: int = 40
    MINI_BATCH: int = 1
    DELTA_L: float = 0.6
    DELTA_G: float = 0.4
    TOTAL_ROUNDS: int = 6000
    BLOCK_SIZE: int = 800
    small_amount: float = 1e-7
    LAMBDA: float = 0.01
    KNN: int = 5
    EPSILON: list = field(default_factory=list)
    DYNAMIC_EPSILON: bool = False
    EG_ETA: float = 0.01
    EG_SQRT_DECAY: bool = True
    EG_STEPS: int = 10
    ALPHA_FLOOR: float = 0.0
    feature_extractor: str = "clip"
    DEVICE: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    DEBUG: bool = True
    CHANGE_POINT: bool = False
    CHANGE_POINT_PATH: str = (
        "/research/d7/rshr/bahar/data/datasets/synthetic_change_point.npz"
    )
    MALE_FEMALE_ADAPTIVE: bool = False
    MALE_FEMALE_DATA_PATH: str = "/research/d7/rshr/bahar/data/datasets/mnist/dino"
    MALE_FEMALE_CHANGE_ROUND: int = 1000
    ARM1_MALE_RATIO_BEFORE: float = 0.0
    ARM1_MALE_RATIO_AFTER: float = 0.5
    ARM2_MALE_RATIO: float = 0.25
    CATS_BIRDS_ADAPTIVE: bool = False
    CATS_BIRDS_DATA_PATH: str = (
        "/research/d7/rshr/bahar/data/dgm/imagenet_groundtruth/"
        "ImageNet-Datasets-Downloader/imagenet"
    )
    CATS_BIRDS_CHANGE_ROUND: int = 3000


class ConfigFID:
    """Configuration for FID-based online evaluation."""

    INITIAL_SAMPLE_COUNT: int = 5
    MINI_BATCH: int = 1
    TOTAL_ROUNDS: int = 10000
    feature_extractor: str = "dino"

    EG_ETA: float = 0.01
    EG_SQRT_DECAY: bool = True
    ALPHA_FLOOR: float = 0.0
    EG_STEPS: int = 1
    OPTIMIZER: str = "scipy"
    SCIPY_METHOD: str = "trust-constr"
    SCIPY_MAXITER: int = 200
    SCIPY_GTOL: float = 1e-10
    SCIPY_XTOL: float = 1e-12
    SCIPY_BARRIER_TOL: float = 1e-12
    SCIPY_FTOL: float = 1e-12
    SCIPY_RESTARTS: int = 2

    FID_EPS: float = 1e-6
    MIN_FID_SAMPLES_PER_ARM: int = 5
    TRACK_MIXTURE_FID: bool = False
    MIXTURE_FID_SAMPLES: int = 10000
    MIXTURE_FID_SEED: int = 42
    EPSILON: float = 0.1
    EPSILON_DECAY: float = 0.0

    CHANGE_POINT: bool = False
    CHANGE_POINT_PATH: str = (
        "/research/d7/rshr/bahar/data/datasets/synthetic_change_point.npz"
    )
    MALE_FEMALE_ADAPTIVE: bool = False
    MALE_FEMALE_DATA_PATH: str = "/research/d7/rshr/bahar/data/datasets/mnist/dino"
    MALE_FEMALE_CHANGE_ROUND: int = 1000
    ARM1_MALE_RATIO_BEFORE: float = 0.0
    ARM1_MALE_RATIO_AFTER: float = 0.5
    ARM2_MALE_RATIO: float = 0.25
    CATS_BIRDS_ADAPTIVE: bool = False
    CATS_BIRDS_DATA_PATH: str = (
        "/research/d7/rshr/bahar/data/dgm/imagenet_groundtruth/"
        "ImageNet-Datasets-Downloader/imagenet"
    )
    CATS_BIRDS_CHANGE_ROUND: int = 3000


class ConfigVNE:
    """Configuration for VNE-based online evaluation."""

    INITIAL_SAMPLE_COUNT: int = 10
    MINI_BATCH: int = 2
    TOTAL_ROUNDS: int = 10000
    feature_extractor: str = "dino"
    rff_n: int = 256
    rff_sigma: float = 40.0
    rff_seed: int = 42
    kernel_type: str = "cosine"

    EG_ETA: float = 0.1
    EG_SQRT_DECAY: bool = True
    ALPHA_FLOOR: float = 0.0

    VNE_EPS: float = 1e-10
    MIN_VNE_SAMPLES: int = 10
    EG_STEPS_PER_ROUND: int = 10
    EPSILON: float = 0.1
    EPSILON_DECAY: float = 0.0


__all__ = ["Config", "ConfigFID", "ConfigVNE"]
