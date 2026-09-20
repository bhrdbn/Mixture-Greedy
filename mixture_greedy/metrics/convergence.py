"""Mixture-weight convergence diagnostics."""

import numpy as np


def alpha_convergence(alpha_hist, alpha_star, eps=1e-12):
    """Return legacy L1, L2, and KL convergence histories."""
    l1 = np.sum(np.abs(alpha_hist - alpha_star), axis=1)
    l2 = np.sqrt(np.sum((alpha_hist - alpha_star) ** 2, axis=1))

    a_star = np.clip(alpha_star, eps, 1.0)
    KL = []
    for alpha in alpha_hist:
        a_t = np.clip(alpha, eps, 1.0)
        KL.append(np.sum(a_star * np.log(a_star / a_t)))
    KL = np.array(KL)
    return l1, l2, KL


__all__ = ["alpha_convergence"]
