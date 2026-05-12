# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Canonical correlation analysis. 'No man ever steps in the same river twice. -- Heraclitus'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def canonical_correlation(
    X: np.ndarray,
    Y: np.ndarray,
) -> DescriptiveResult:
    """Canonical Correlation Analysis (CCA).

    Finds linear combinations of X and Y that maximise
    correlation.  Returns canonical correlations and loadings.

    :param X: (n, p) first set of variables.
    :param Y: (n, q) second set of variables.
    :return: DescriptiveResult with canonical correlations.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError("X and Y must have the same number of rows.")

    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)

    Sxx = (Xc.T @ Xc) / (n - 1)
    Syy = (Yc.T @ Yc) / (n - 1)
    Sxy = (Xc.T @ Yc) / (n - 1)

    def _mat_sqrt_inv(M):
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 1e-12)
        return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    Sxx_inv_half = _mat_sqrt_inv(Sxx)
    Syy_inv_half = _mat_sqrt_inv(Syy)

    M = Sxx_inv_half @ Sxy @ Syy_inv_half
    U, s, Vt = np.linalg.svd(M, full_matrices=False)

    canon_corrs = np.clip(s, 0.0, 1.0)
    x_loadings = Sxx_inv_half @ U
    y_loadings = Syy_inv_half @ Vt.T

    return DescriptiveResult(
        name="cancc",
        value=float(canon_corrs[0]) if len(canon_corrs) > 0 else 0.0,
        extra={
            "canonical_correlations": canon_corrs.tolist(),
            "n": n,
            "p": X.shape[1],
            "q": Y.shape[1],
            "x_loadings": x_loadings.tolist(),
            "y_loadings": y_loadings.tolist(),
        },
    )


cancc = canonical_correlation


def cheatsheet() -> str:
    return "He who would learn to fly one day must first learn to stand and walk. -- Friedrich Nietzsche"
