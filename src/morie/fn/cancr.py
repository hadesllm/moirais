# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Canonical correlation analysis."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def canonical_correlation(
    X: np.ndarray,
    Y: np.ndarray,
) -> DescriptiveResult:
    """Canonical Correlation Analysis (CCA).

    Parameters
    ----------
    X : ndarray (n, p)
        First set of variables.
    Y : ndarray (n, q)
        Second set of variables.

    Returns
    -------
    DescriptiveResult
        ``value`` is the array of canonical correlations.
        ``extra`` has ``X_weights``, ``Y_weights``, ``X_scores``, ``Y_scores``.
    """
    Xa = np.asarray(X, dtype=np.float64)
    Ya = np.asarray(Y, dtype=np.float64)
    n = Xa.shape[0]
    Xa = Xa - Xa.mean(axis=0)
    Ya = Ya - Ya.mean(axis=0)

    p = Xa.shape[1]
    q = Ya.shape[1]
    k = min(p, q)

    Sxx = Xa.T @ Xa / (n - 1) + np.eye(p) * 1e-8
    Syy = Ya.T @ Ya / (n - 1) + np.eye(q) * 1e-8
    Sxy = Xa.T @ Ya / (n - 1)

    Sxx_inv_half = _matrix_power(Sxx, -0.5)
    Syy_inv_half = _matrix_power(Syy, -0.5)

    M = Sxx_inv_half @ Sxy @ Syy_inv_half
    U, s, Vt = np.linalg.svd(M, full_matrices=False)

    correlations = s[:k]
    A = Sxx_inv_half @ U[:, :k]
    B = Syy_inv_half @ Vt[:k].T

    X_scores = Xa @ A
    Y_scores = Ya @ B

    return DescriptiveResult(
        name="CCA",
        value=correlations,
        extra={
            "X_weights": A,
            "Y_weights": B,
            "X_scores": X_scores,
            "Y_scores": Y_scores,
            "n_components": k,
        },
    )


def _matrix_power(A: np.ndarray, power: float) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 1e-12)
    return eigvecs @ np.diag(eigvals ** power) @ eigvecs.T


cancr = canonical_correlation


def cheatsheet() -> str:
    return "canonical_correlation({}) -> Canonical correlation analysis."
