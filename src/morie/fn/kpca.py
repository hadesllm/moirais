# morie.fn -- function file (hadesllm/morie)
"""Kernel PCA."""

from __future__ import annotations

import numpy as np

from ._containers import PcaRes


def kernel_pca(
    data: np.ndarray,
    n_components: int = 2,
    kernel: str = "rbf",
    gamma: float | None = None,
    degree: int = 3,
) -> PcaRes:
    """Kernel Principal Component Analysis.

    Parameters
    ----------
    data : ndarray (n, p)
        Data matrix.
    n_components : int
        Number of components.
    kernel : str
        ``'rbf'``, ``'poly'``, or ``'linear'``.
    gamma : float, optional
        RBF bandwidth. Defaults to 1/p.
    degree : int
        Polynomial degree (only for ``'poly'``).

    Returns
    -------
    PcaRes
    """
    X = np.asarray(data, dtype=np.float64)
    n, p = X.shape
    if gamma is None:
        gamma = 1.0 / p

    if kernel == "rbf":
        sq = np.sum(X ** 2, axis=1)
        K = np.exp(-gamma * (sq[:, None] + sq[None, :] - 2 * X @ X.T))
    elif kernel == "poly":
        K = (gamma * X @ X.T + 1) ** degree
    else:
        K = X @ X.T

    ones_n = np.ones((n, n)) / n
    K_centered = K - ones_n @ K - K @ ones_n + ones_n @ K @ ones_n

    eigvals, eigvecs = np.linalg.eigh(K_centered)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    k = min(n_components, n)
    eigvals_k = np.maximum(eigvals[:k], 0)
    alpha = eigvecs[:, :k]
    for i in range(k):
        if eigvals_k[i] > 0:
            alpha[:, i] /= np.sqrt(eigvals_k[i])

    scores = K_centered @ alpha
    total_var = np.sum(np.maximum(eigvals, 0))
    ratio = eigvals_k / total_var if total_var > 0 else np.zeros(k)

    return PcaRes(
        components=alpha,
        explained_variance=eigvals_k,
        explained_variance_ratio=ratio,
        scores=scores,
        n=n,
        p=p,
    )


kpca = kernel_pca


def cheatsheet() -> str:
    return "kernel_pca({}) -> Kernel PCA (RBF/poly/linear)."
