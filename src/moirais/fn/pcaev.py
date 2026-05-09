# moirais.fn — function file (hadesllm/moirais)
"""PCA with eigenvalue decomposition."""

from __future__ import annotations

import numpy as np

from ._containers import PcaRes


def pca_eigen(
    data: np.ndarray,
    n_components: int | None = None,
    center: bool = True,
    scale: bool = False,
) -> PcaRes:
    """Principal Component Analysis via eigenvalue decomposition of the covariance matrix.

    Parameters
    ----------
    data : ndarray (n, p)
        Data matrix with observations in rows.
    n_components : int, optional
        Number of components to retain. Defaults to min(n, p).
    center : bool
        Mean-center the data.
    scale : bool
        Standardise columns to unit variance.

    Returns
    -------
    PcaRes
        ``components`` (loadings), ``explained_variance`` (eigenvalues),
        ``explained_variance_ratio``, ``scores``.
    """
    X = np.asarray(data, dtype=np.float64)
    n, p = X.shape
    if center:
        X = X - X.mean(axis=0)
    if scale:
        s = X.std(axis=0, ddof=0)
        s[s == 0] = 1.0
        X = X / s

    cov = np.cov(X, rowvar=False, ddof=1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    k = n_components or min(n, p)
    eigvals_k = eigvals[:k]
    eigvecs_k = eigvecs[:, :k]
    total_var = eigvals.sum()
    ratio = eigvals_k / total_var if total_var > 0 else np.zeros(k)
    scores = X @ eigvecs_k

    return PcaRes(
        components=eigvecs_k,
        explained_variance=eigvals_k,
        explained_variance_ratio=ratio,
        scores=scores,
        n=n,
        p=p,
    )


pcaev = pca_eigen


def cheatsheet() -> str:
    return "pca_eigen({}) -> PCA via eigenvalue decomposition of covariance matrix."
