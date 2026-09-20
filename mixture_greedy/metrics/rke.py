"""RKE/KID metric helpers shared by the offline and online evaluators."""

import numpy as np
import torch

from mixture_greedy.config import Config
from mixture_greedy.metrics.distances import compute_pairwise_distance
from mixture_greedy.metrics.fid import compute_fid

# One shared default instance is used by every RKE module for compatibility.
default_config = Config()


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_reals_nearest_neighbour_distances(real_features, nearest_k):
    return compute_nearest_neighbour_distances(real_features, nearest_k)


def compute_linear_term(
    real_features, fake_features, nearest_k, reals_nnd=None, config=None
):
    if config is None:
        config = default_config
    if config.LINEAR_METRIC == "kid":
        kernel = KernelUtils.frobenius_norm(fake_features, real_features, config=config)
        kernel /= len(real_features) * len(fake_features)
        return 2 * kernel
    if reals_nnd is None:
        real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
            real_features, nearest_k
        )
    else:
        real_nearest_neighbour_distances = reals_nnd
    distance_real_fake = compute_pairwise_distance(real_features, fake_features)
    if config.LINEAR_METRIC == "precision":
        prec = (
            distance_real_fake
            < np.expand_dims(real_nearest_neighbour_distances, axis=1)
        ).any(axis=0).mean()
        return prec
    if config.LINEAR_METRIC == "density":
        return (1.0 / float(nearest_k)) * (
            distance_real_fake
            < np.expand_dims(real_nearest_neighbour_distances, axis=1)
        ).sum(axis=0).mean()
    print("The linear metric is invalid")
    assert 1 == 0


def _linear_coeff(config):
    # For KID, the mandatory cross term should not be scaled by LAMBDA.
    if config.QUADRATIC_METRIC == "kid":
        return 1.0
    return config.LAMBDA


class KernelUtils:
    @staticmethod
    def gaussian_kernel(x, y, sigma):
        dist_sq = torch.sum((x - y) ** 2, dim=-1)
        return torch.exp(-0.5 * dist_sq / sigma**2)

    @staticmethod
    def frobenius_norm(X, Y, sigma=None, block_size=None, config=None):
        if config is None:
            config = default_config
        if sigma is None:
            sigma = config.SIGMA
        if block_size is None:
            block_size = config.BLOCK_SIZE
        is_renyi = True

        if config.QUADRATIC_METRIC == "kid":
            is_renyi = False
        X, Y = torch.Tensor(X).to(config.DEVICE), torch.Tensor(Y).to(config.DEVICE)
        n_data_x, n_data_y = X.size(0), Y.size(0)
        sum_frobenius = 0.0

        for i in range(0, n_data_x, block_size):
            for j in range(0, n_data_y, block_size):
                X_block = X[i : i + block_size]
                Y_block = Y[j : j + block_size]
                if is_renyi:
                    kernel_block = KernelUtils.gaussian_kernel(
                        X_block.unsqueeze(0), Y_block.unsqueeze(1), sigma
                    ) ** 2
                else:
                    kernel_block = KernelUtils.gaussian_kernel(
                        X_block.unsqueeze(0), Y_block.unsqueeze(1), sigma
                    )
                sum_frobenius += torch.sum(kernel_block).item()

        return sum_frobenius

    @staticmethod
    def scaled_kernel(kernel, sizes, config=None):
        if config is None:
            config = default_config
        sizes = 1 / np.array(sizes)
        size_matrix = sizes.reshape(sizes.shape[-1], 1) @ sizes.reshape(
            1, sizes.shape[-1]
        )
        if config.DEBUG:
            print("size matrix", size_matrix.shape)
        kernel_resized = size_matrix * kernel
        if config.QUADRATIC_METRIC == "kid" and config.KID_NEW:
            print(kernel_resized.shape)
            for i in range(len(sizes)):
                kernel_resized[i, i] = kernel[i, i] / (sizes[i] * (sizes[i] - 1))
        if config.DEBUG:
            print("kernel resized size", kernel_resized)
        return kernel_resized


__all__ = [
    "KernelUtils",
    "compute_fid",
    "compute_linear_term",
    "compute_nearest_neighbour_distances",
    "compute_reals_nearest_neighbour_distances",
    "default_config",
    "get_kth_value",
]
