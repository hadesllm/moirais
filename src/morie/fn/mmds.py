# morie.fn -- function file (hadesllm/morie)
"""Metric MDS for ideal point estimation (Armstrong Ch 3)."""

from __future__ import annotations

import numpy as np

from ._containers import MdsRes


def metric_mds(dissimilarities, n_dims: int = 2, *, max_iter: int = 300, tol: float = 1e-6) -> MdsRes:
    """Metric MDS via SMACOF for ideal point recovery.

    Minimises raw stress: sum_{i<j} (d_ij - delta_ij)^2 where delta are
    the observed dissimilarities and d are embedded distances.

    :param dissimilarities: Square symmetric dissimilarity matrix.
    :param n_dims: Embedding dimensions.
    :param max_iter: Maximum SMACOF iterations.
    :param tol: Convergence tolerance on stress change.
    :return: MdsRes with coordinates and stress.

    References
    ----------
    Armstrong, D. A. (2014). Analyzing Spatial Models of Choice and Judgment.
        CRC Press. Chapter 3.

    .. epigraph:: Knowledge is power. -- Francis Bacon
    """
    D = np.asarray(dissimilarities, dtype=float)
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError("dissimilarities must be a square matrix.")

    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, n_dims))

    def _distances(X):
        diff = X[:, None, :] - X[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1) + 1e-14)

    def _stress(X):
        d = _distances(X)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        return np.sqrt(((d[mask] - D[mask]) ** 2).sum() / max((D[mask] ** 2).sum(), 1e-14))

    stress = _stress(X)
    for _ in range(max_iter):
        d = _distances(X)
        B = np.zeros_like(D)
        nonzero = d > 1e-14
        B[nonzero] = -D[nonzero] / d[nonzero]
        np.fill_diagonal(B, 0.0)
        np.fill_diagonal(B, -B.sum(axis=1))
        X_new = B @ X / n
        new_stress = _stress(X_new)
        if abs(stress - new_stress) < tol:
            X = X_new
            stress = new_stress
            break
        X = X_new
        stress = new_stress

    H = np.eye(n) - np.ones((n, n)) / n
    B_dc = -0.5 * H @ (D ** 2) @ H
    eigvals = np.linalg.eigvalsh(B_dc)[::-1]

    return MdsRes(coordinates=X, stress=stress, eigenvalues=eigvals[:n_dims])


mmds = metric_mds


def cheatsheet() -> str:
    return "metric_mds({}) -> Metric MDS for ideal point estimation."
