# moirais.fn — function file (hadesllm/moirais)
"""PCA via singular value decomposition."""

from __future__ import annotations

import numpy as np

from ._containers import PcaRes


def pca_svd(
    data: np.ndarray,
    n_components: int | None = None,
    center: bool = True,
    scale: bool = False,
) -> PcaRes:
    """Principal Component Analysis via SVD of the centred data matrix.

    Parameters
    ----------
    data : ndarray (n, p)
        Data matrix.
    n_components : int, optional
        Number of components to retain.
    center : bool
        Mean-center columns.
    scale : bool
        Standardise columns to unit variance.

    Returns
    -------
    PcaRes
    """
    X = np.asarray(data, dtype=np.float64)
    n, p = X.shape
    if center:
        X = X - X.mean(axis=0)
    if scale:
        s = X.std(axis=0, ddof=0)
        s[s == 0] = 1.0
        X = X / s

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    eigvals = S ** 2 / (n - 1)

    k = n_components or min(n, p)
    components = Vt[:k].T
    eigvals_k = eigvals[:k]
    total_var = eigvals.sum()
    ratio = eigvals_k / total_var if total_var > 0 else np.zeros(k)
    scores = X @ components

    return PcaRes(
        components=components,
        explained_variance=eigvals_k,
        explained_variance_ratio=ratio,
        scores=scores,
        n=n,
        p=p,
    )


pcsvd = pca_svd


def cheatsheet() -> str:
    return "pca_svd({}) -> PCA via singular value decomposition."
