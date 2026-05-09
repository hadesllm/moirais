# moirais.fn — function file (hadesllm/moirais)
"""Projection pursuit."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def projection_pursuit(
    data: np.ndarray,
    n_components: int = 2,
    index: str = "kurtosis",
    max_iter: int = 200,
    seed: int = 42,
) -> DescriptiveResult:
    """Projection pursuit: find projections maximising a projection index.

    Parameters
    ----------
    data : ndarray (n, p)
        Data matrix (centred internally).
    n_components : int
        Number of projection directions.
    index : str
        ``'kurtosis'`` or ``'negentropy'``.
    max_iter : int
        Maximum iterations per direction.
    seed : int
        Random seed.

    Returns
    -------
    DescriptiveResult
        ``value`` is the projected data (n x n_components).
        ``extra`` has ``directions`` and ``index_values``.
    """
    X = np.asarray(data, dtype=np.float64)
    n, p = X.shape
    X = X - X.mean(axis=0)
    cov = np.cov(X, rowvar=False, ddof=1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    safe = np.maximum(eigvals, 1e-12)
    W = eigvecs @ np.diag(1.0 / np.sqrt(safe)) @ eigvecs.T
    Z = X @ W

    rng = np.random.default_rng(seed)
    directions = np.zeros((p, n_components))
    index_values = []

    for comp in range(n_components):
        a = rng.standard_normal(p)
        a /= np.linalg.norm(a)

        for _ in range(max_iter):
            proj = Z @ a
            if index == "kurtosis":
                idx_val = float(np.mean(proj ** 4) - 3)
                grad = 4 * Z.T @ (proj ** 3) / n
            else:
                idx_val = float(np.mean(-np.exp(-proj ** 2 / 2)))
                grad = Z.T @ (proj * np.exp(-proj ** 2 / 2)) / n

            a_new = grad.copy()
            for j in range(comp):
                a_new -= np.dot(a_new, directions[:, j]) * directions[:, j]
            norm = np.linalg.norm(a_new)
            if norm < 1e-12:
                break
            a_new /= norm
            if np.abs(np.dot(a_new, a)) > 1.0 - 1e-8:
                a = a_new
                break
            a = a_new

        directions[:, comp] = a
        index_values.append(idx_val)

    projected = Z @ directions
    return DescriptiveResult(
        name="ProjectionPursuit",
        value=projected,
        extra={
            "directions": directions,
            "index_values": index_values,
            "index": index,
        },
    )


ppurs = projection_pursuit


def cheatsheet() -> str:
    return "projection_pursuit({}) -> Projection pursuit via kurtosis/negentropy."
