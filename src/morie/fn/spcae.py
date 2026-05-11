"""Sparse PCA via iterative thresholding."""

from __future__ import annotations

import numpy as np

from ._containers import PcaRes


def sparse_pca(
    data: np.ndarray,
    n_components: int = 2,
    alpha: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> PcaRes:
    """Sparse PCA via elastic-net-penalised power iteration.

    Uses coordinate descent with L1 penalty to produce sparse loadings.

    Parameters
    ----------
    data : ndarray (n, p)
        Data matrix (will be centred).
    n_components : int
        Number of sparse components.
    alpha : float
        L1 penalty strength.
    max_iter : int
        Maximum iterations per component.
    tol : float
        Convergence tolerance.

    Returns
    -------
    PcaRes
    """
    X = np.asarray(data, dtype=np.float64)
    n, p = X.shape
    X = X - X.mean(axis=0)

    components = np.zeros((p, n_components))
    R = X.copy()

    for comp in range(n_components):
        cov = R.T @ R / (n - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        v = eigvecs[:, -1].copy()

        z_probe = R.T @ (R @ v) / (n - 1)
        max_z = float(np.max(np.abs(z_probe)))
        effective_alpha = min(alpha, 0.9 * max_z) if max_z > 0 else 0.0

        for _ in range(max_iter):
            z = R.T @ (R @ v) / (n - 1)
            v_new = np.sign(z) * np.maximum(np.abs(z) - effective_alpha, 0)
            norm = np.linalg.norm(v_new)
            if norm < 1e-12:
                v = np.zeros_like(v)
                break
            v_new /= norm
            if np.linalg.norm(v_new - v) < tol:
                v = v_new
                break
            v = v_new

        components[:, comp] = v
        R = R - (R @ v)[:, None] * v[None, :]

    scores = X @ components
    var_explained = np.var(scores, axis=0, ddof=1)
    total_var = np.sum(np.var(X, axis=0, ddof=1))
    ratio = var_explained / total_var if total_var > 0 else np.zeros(n_components)

    return PcaRes(
        components=components,
        explained_variance=var_explained,
        explained_variance_ratio=ratio,
        scores=scores,
        n=n,
        p=p,
    )


spcae = sparse_pca


def cheatsheet() -> str:
    return "sparse_pca({}) -> Sparse PCA via iterative thresholding."
