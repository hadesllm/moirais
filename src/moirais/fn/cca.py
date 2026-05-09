# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Canonical correlation analysis."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def canonical_correlation(X: np.ndarray, Y: np.ndarray) -> DescriptiveResult:
    """CCA via SVD of the cross-covariance matrix.

    Parameters
    ----------
    X : (n, p) array
    Y : (n, q) array

    Returns
    -------
    DescriptiveResult
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError("X and Y must have same n.")
    p, q = X.shape[1], Y.shape[1]

    X_c = X - X.mean(axis=0)
    Y_c = Y - Y.mean(axis=0)

    Sxx = X_c.T @ X_c / (n - 1)
    Syy = Y_c.T @ Y_c / (n - 1)
    Sxy = X_c.T @ Y_c / (n - 1)

    Sxx_inv_half = np.linalg.inv(np.linalg.cholesky(Sxx + 1e-8 * np.eye(p)))
    Syy_inv_half = np.linalg.inv(np.linalg.cholesky(Syy + 1e-8 * np.eye(q)))

    M = Sxx_inv_half @ Sxy @ Syy_inv_half
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    cancorr = np.clip(s, 0, 1)

    return DescriptiveResult(
        name="cca",
        value=float(cancorr[0]) if len(cancorr) > 0 else 0.0,
        extra={"canonical_correlations": cancorr.tolist(), "n": n, "p": p, "q": q},
    )


cca = canonical_correlation


def cheatsheet() -> str:
    return "canonical_correlation({}) -> Canonical correlation analysis."
