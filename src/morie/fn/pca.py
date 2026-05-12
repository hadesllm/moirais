# morie.fn -- function file (hadesllm/morie)
"""Principal Component Analysis via eigendecomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._containers import PcaRes


def pca(
    data: pd.DataFrame | np.ndarray,
    n_components: int | None = None,
) -> PcaRes:
    """Principal Component Analysis.

    Standardises columns to zero mean / unit variance, computes the
    covariance matrix, and eigendecomposes it.

    Parameters
    ----------
    data : DataFrame or ndarray
        Observations as rows, variables as columns.
    n_components : int, optional
        Number of components to retain.  *None* keeps all.

    Returns
    -------
    PcaRes
        ``components`` (loadings, p x k), ``explained_variance`` (eigenvalues),
        ``explained_variance_ratio``, ``scores`` (transformed data, n x k).

    References
    ----------
    Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.).
    Springer.  DOI: 10.1007/b98835
    """
    X = np.asarray(data, dtype=np.float64)
    n, p = X.shape

    # Standardise
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd

    # Covariance (n-1 normalisation)
    cov = np.cov(Z, rowvar=False)

    # Eigendecomposition (symmetric -> eigh is numerically preferable)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Truncate
    k = n_components if n_components is not None else p
    k = min(k, p)
    eigvals_k = eigvals[:k]
    components = eigvecs[:, :k]

    total_var = np.sum(eigvals)
    ratio = eigvals_k / total_var if total_var > 0 else np.zeros(k)

    scores = Z @ components

    return PcaRes(
        components=components,
        explained_variance=eigvals_k,
        explained_variance_ratio=ratio,
        scores=scores,
        n=n,
        p=p,
    )


def cheatsheet() -> str:
    return "pca({}) -> Principal Component Analysis via eigendecomposition."
