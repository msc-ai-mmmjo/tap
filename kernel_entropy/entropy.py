"""
Kernel Language Entropy calculation.

Transforms similarity matrix W into Von Neumann Entropy through:
W -> Laplacian -> Heat Kernel -> Density Matrix -> VNE
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import torch to avoid CUDA context conflict with llama-cpp-python.
# PyTorch must be imported AFTER llama-cpp-python initializes CUDA or ELSE...
if TYPE_CHECKING:
    import torch


def kle_from_similarity(W: "torch.Tensor", t: float = 1.0) -> float:
    """
    Compute Kernel Language Entropy from similarity matrix.

    Args:
        W: N×N symmetric similarity matrix on CUDA (from NLI scoring)
        t: Heat kernel lengthscale (default: 1.0)

    Returns:
        Von Neumann Entropy (float)
    """
    import torch  # Lazy import to avoid CUDA conflict with llama-cpp-python

    n = W.shape[0]

    # Edge cases
    if n < 2:
        return 0.0

    # L = D - W (unnormalized Laplacian)
    D = torch.diag(W.sum(dim=1))
    L = D - W

    # K = exp(-t * L) using matrix exponential
    K = torch.linalg.matrix_exp(-t * L)

    # Normalize by diagonal: K̃[i,j] = K[i,j] / sqrt(K[i,i] * K[j,j])
    # This makes diag(K̃) = 1, so trace(K̃) = n. Divide by trace for unit trace density matrix.
    diag_sqrt = torch.sqrt(torch.diag(K))
    K_norm = K / torch.outer(diag_sqrt, diag_sqrt)
    rho = K_norm / n

    # VNE = -Σ λᵢ × ln(λᵢ) for λᵢ > threshold
    eigenvalues = torch.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    vne = -torch.sum(eigenvalues * torch.log(eigenvalues))

    return vne.item()
