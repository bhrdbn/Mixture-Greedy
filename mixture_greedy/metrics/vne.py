"""Von Neumann entropy helpers for weighted model mixtures."""

import torch


def compute_mixture_vne(rho_list: list, alpha: torch.Tensor, eps: float = 1e-10):
    """Compute the Von Neumann entropy of a weighted density-matrix mixture."""
    # These local variables are retained because the legacy implementation
    # evaluated them even though they are not subsequently read.
    _ = alpha.device, alpha.dtype

    rho_mixture = torch.zeros_like(rho_list[0])
    for index, rho in enumerate(rho_list):
        rho_mixture = rho_mixture + alpha[index] * rho

    rho_mixture = 0.5 * (rho_mixture + rho_mixture.T)
    eigenvalues = torch.linalg.eigvalsh(rho_mixture)
    eigenvalues = torch.clamp(eigenvalues, min=eps)
    eigenvalues = eigenvalues / eigenvalues.sum()
    return -torch.sum(eigenvalues * torch.log(eigenvalues + eps))


mixture_vne = compute_mixture_vne

__all__ = ["mixture_vne"]
