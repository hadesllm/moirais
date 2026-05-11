# morie.fn — function file (hadesllm/morie)
"""Compare factor analysis vs PCA solutions."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def factor_pca_compare(
    X: np.ndarray,
    n_components: int = 2,
) -> DescriptiveResult:
    """Compare PCA eigenvalues with FA communalities.

    Parameters
    ----------
    X : (n, p) array
    n_components : int

    Returns
    -------
    DescriptiveResult
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    q = min(n_components, p)

    R = np.corrcoef(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(R)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx][:q]
    eigvecs = eigvecs[:, idx][:, :q]

    pca_var = eigvals / p
    loadings = eigvecs * np.sqrt(np.maximum(eigvals, 0))[None, :]
    communalities = np.sum(loadings**2, axis=1)
    uniquenesses = 1 - communalities

    return DescriptiveResult(
        name="fa_pca_compare",
        value=float(np.sum(pca_var)),
        extra={
            "pca_eigenvalues": eigvals.tolist(),
            "pca_var_explained": pca_var.tolist(),
            "communalities": communalities.tolist(),
            "uniquenesses": uniquenesses.tolist(),
            "n_components": q,
            "n": n,
            "p": p,
        },
    )


fapca = factor_pca_compare


def cheatsheet() -> str:
    return "factor_pca_compare({}) -> Compare factor analysis vs PCA solutions."
