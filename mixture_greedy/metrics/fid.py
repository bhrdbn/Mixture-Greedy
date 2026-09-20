"""Differentiable Fréchet distance helpers used by the FID evaluator."""

import numpy as np
import torch


def compute_fid(real_features, fake_features, eps=1e-6, device=None):
    """Compute sample FID using the legacy float32 implementation."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_t = torch.from_numpy(real_features).float().to(device)
    fake_t = torch.from_numpy(fake_features).float().to(device)

    mu_real = real_t.mean(dim=0)
    real_centered = real_t - mu_real
    sigma_real = (real_centered.T @ real_centered) / (real_t.shape[0] - 1)

    mu_fake = fake_t.mean(dim=0)
    fake_centered = fake_t - mu_fake
    sigma_fake = (fake_centered.T @ fake_centered) / (fake_t.shape[0] - 1)

    diff = mu_real - mu_fake
    diff_squared = torch.dot(diff, diff).item()

    identity = torch.eye(sigma_real.shape[0], device=device)
    sigma_real = sigma_real + eps * identity
    sigma_fake = sigma_fake + eps * identity

    eigenvalues_real, eigenvectors_real = torch.linalg.eigh(sigma_real)
    eigenvalues_real = torch.clamp(eigenvalues_real, min=0)
    sqrt_sigma_real = (
        eigenvectors_real
        @ torch.diag(torch.sqrt(eigenvalues_real))
        @ eigenvectors_real.T
    )

    symmetric_product = sqrt_sigma_real @ sigma_fake @ sqrt_sigma_real
    eigenvalues_product, _ = torch.linalg.eigh(symmetric_product)
    eigenvalues_product = torch.clamp(eigenvalues_product, min=0)

    trace_covariance_mean = torch.sum(torch.sqrt(eigenvalues_product)).item()
    fid = (
        diff_squared
        + torch.trace(sigma_real).item()
        + torch.trace(sigma_fake).item()
        - 2 * trace_covariance_mean
    )
    return max(0.0, fid)


def _mean_cov_np(X: np.ndarray, eps=1e-6):
    """Return the legacy MLE mean and regularized covariance estimate."""
    X = np.asarray(X)
    mu = X.mean(axis=0)
    Xm = X - mu
    cov = (Xm.T @ Xm) / max(X.shape[0], 1)
    cov = 0.5 * (cov + cov.T)
    cov = cov + eps * np.eye(cov.shape[0])
    return mu, cov


def _mixture_moments_torch(alpha, mus, covs):
    """Return first and second moments for a weighted Gaussian mixture."""
    mu = (alpha[:, None] * mus).sum(dim=0)
    second = (alpha[:, None, None] * (covs + mus[:, :, None] @ mus[:, None, :])).sum(
        dim=0
    )
    cov = second - (mu[:, None] @ mu[None, :])
    cov = 0.5 * (cov + cov.T)
    return mu, cov


def _trace_sqrtm_psd(A, eps=1e-12):
    """Return the trace of the square root of a symmetric PSD matrix."""
    A = 0.5 * (A + A.T)
    evals, _ = torch.linalg.eigh(A)
    evals = torch.clamp(evals, min=eps)
    return torch.sum(torch.sqrt(evals))


def fid_from_moments_torch(mu_g, cov_g, mu_r, cov_r, eps=1e-6):
    """Compute differentiable FID between two Gaussian distributions."""
    dimensions = cov_g.shape[0]
    identity = torch.eye(dimensions, device=cov_g.device, dtype=cov_g.dtype)

    cov_g = 0.5 * (cov_g + cov_g.T) + eps * identity
    cov_r = 0.5 * (cov_r + cov_r.T) + eps * identity

    diff = mu_g - mu_r
    diff_term = diff @ diff

    evals_r, evecs_r = torch.linalg.eigh(cov_r)
    evals_r = torch.clamp(evals_r, min=eps)
    sqrt_r = evecs_r @ torch.diag(torch.sqrt(evals_r)) @ evecs_r.T

    A = sqrt_r @ cov_g @ sqrt_r
    trace_sqrt = _trace_sqrtm_psd(A, eps=eps)
    return diff_term + torch.trace(cov_g) + torch.trace(cov_r) - 2.0 * trace_sqrt


def _fid_value_and_grad_for_alpha(alpha_np, mus, covs, mu_r, cov_r, eps=1e-6):
    """Return FID and its gradient with respect to mixture weights."""
    alpha_t = torch.tensor(alpha_np, dtype=torch.float64, requires_grad=True)
    mu_mix, cov_mix = _mixture_moments_torch(alpha_t, mus, covs)
    fid_obj = fid_from_moments_torch(mu_mix, cov_mix, mu_r, cov_r, eps=eps)
    fid_obj.backward()
    value = float(fid_obj.detach().cpu().item())
    grad = alpha_t.grad.detach().cpu().numpy().astype(np.float64)
    return value, grad


# Descriptive public aliases for new package users.
mean_and_covariance = _mean_cov_np
mixture_moments = _mixture_moments_torch
trace_sqrt_psd = _trace_sqrtm_psd
fid_from_moments = fid_from_moments_torch
fid_value_and_gradient = _fid_value_and_grad_for_alpha

__all__ = [
    "compute_fid",
    "fid_from_moments",
    "fid_value_and_gradient",
    "mean_and_covariance",
    "mixture_moments",
    "trace_sqrt_psd",
]
