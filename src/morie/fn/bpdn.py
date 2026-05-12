# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Basis Pursuit Denoising via L1 minimization."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Your focus determines your reality."


def basis_pursuit(D, x, lambda_: float = 0.1, max_iter: int = 500, tol: float = 1e-6, **kwargs) -> DescriptiveResult:
    """Basis Pursuit Denoising via ISTA (Iterative Shrinkage-Thresholding).

    Solves min_c 0.5*||x - D c||^2 + lambda*||c||_1.

    Parameters
    ----------
    D : array-like, shape (n, m)
        Dictionary matrix.
    x : array-like, shape (n,)
        Signal to approximate.
    lambda_ : float
        Regularization parameter (default 0.1).
    max_iter : int
        Maximum iterations (default 500).
    tol : float
        Convergence tolerance (default 1e-6).

    Returns
    -------
    DescriptiveResult
        ``value`` is reconstruction error; ``extra`` has ``coeffs``,
        ``n_nonzero``, ``reconstruction``.

    References
    ----------
    Chen, S. S., Donoho, D. L., & Saunders, M. A. (2001). Atomic
    decomposition by basis pursuit. *SIAM Rev.*, 43(1), 129-159.
    """
    D = np.asarray(D, dtype=float)
    x = np.asarray(x, dtype=float).ravel()
    DtD = D.T @ D
    L = float(np.linalg.norm(DtD, ord=2))
    step = 1.0 / L
    Dtx = D.T @ x
    coeffs = np.zeros(D.shape[1])
    for _ in range(max_iter):
        grad = DtD @ coeffs - Dtx
        z = coeffs - step * grad
        coeffs_new = np.sign(z) * np.maximum(np.abs(z) - lambda_ * step, 0.0)
        if np.linalg.norm(coeffs_new - coeffs) < tol:
            coeffs = coeffs_new
            break
        coeffs = coeffs_new
    reconstruction = D @ coeffs
    err = float(np.linalg.norm(x - reconstruction))
    n_nonzero = int(np.count_nonzero(coeffs))
    return DescriptiveResult(
        name="basis_pursuit",
        value=err,
        extra={"coeffs": coeffs, "n_nonzero": n_nonzero, "reconstruction": reconstruction, "lambda": lambda_},
    )


bpdn = basis_pursuit


def cheatsheet() -> str:
    return "basis_pursuit({}) -> Basis Pursuit Denoising via L1 minimization."
