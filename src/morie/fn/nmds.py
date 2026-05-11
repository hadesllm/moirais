# morie.fn — function file (hadesllm/morie)
"""Non-metric MDS (Kruskal's stress)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from ._containers import MdsRes


def nmds(
    data: np.ndarray,
    n_dims: int = 2,
    max_iter: int = 300,
    seed: int = 42,
) -> MdsRes:
    """Non-metric multidimensional scaling (Kruskal stress minimisation).

    Parameters
    ----------
    data : ndarray
        Either (n, n) distance/dissimilarity matrix or (n, p) data matrix.
        If not square, Euclidean distances are computed.
    n_dims : int
        Number of embedding dimensions.
    max_iter : int
        Maximum SMACOF iterations.
    seed : int
        Random seed for initialisation.

    Returns
    -------
    MdsRes
        ``coordinates``, ``stress``, ``eigenvalues`` (empty for NMDS).
    """
    D = np.asarray(data, dtype=np.float64)
    if D.ndim == 2 and D.shape[0] == D.shape[1]:
        dist = D
    else:
        dist = squareform(pdist(D, metric="euclidean"))

    n = dist.shape[0]
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_dims))

    for _ in range(max_iter):
        d_x = squareform(pdist(X))
        d_x_flat = d_x[np.triu_indices(n, k=1)]
        dist_flat = dist[np.triu_indices(n, k=1)]

        order = np.argsort(dist_flat)
        d_hat = np.zeros_like(d_x_flat)
        d_hat[order] = _isotonic_regression(d_x_flat[order])

        D_hat = np.zeros_like(d_x)
        D_hat[np.triu_indices(n, k=1)] = d_hat
        D_hat += D_hat.T

        B = np.zeros_like(d_x)
        nonzero = d_x > 0
        B[nonzero] = -D_hat[nonzero] / d_x[nonzero]
        np.fill_diagonal(B, -B.sum(axis=1))

        X_new = B @ X / n
        X_new -= X_new.mean(axis=0)

        if np.linalg.norm(X_new - X) / max(np.linalg.norm(X), 1e-12) < 1e-6:
            X = X_new
            break
        X = X_new

    d_final = squareform(pdist(X))
    d_f = d_final[np.triu_indices(n, k=1)]
    dist_f = dist[np.triu_indices(n, k=1)]
    order = np.argsort(dist_f)
    d_hat_f = _isotonic_regression(d_f[order])
    stress = np.sqrt(np.sum((d_f[order] - d_hat_f) ** 2) / max(np.sum(d_f[order] ** 2), 1e-12))

    return MdsRes(
        coordinates=X,
        stress=float(stress),
        eigenvalues=np.array([]),
    )


def _isotonic_regression(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators for isotonic regression."""
    n = len(y)
    result = y.astype(float).copy()
    blocks = [[i] for i in range(n)]

    merged = True
    while merged:
        merged = False
        new_blocks = [blocks[0]]
        for i in range(1, len(blocks)):
            prev_mean = np.mean(result[new_blocks[-1]])
            curr_mean = np.mean(result[blocks[i]])
            if curr_mean < prev_mean:
                combined = new_blocks[-1] + blocks[i]
                m = np.mean(result[combined])
                for idx in combined:
                    result[idx] = m
                new_blocks[-1] = combined
                merged = True
            else:
                new_blocks.append(blocks[i])
        blocks = new_blocks

    return result


nmds_fn = nmds


def cheatsheet() -> str:
    return "nmds({}) -> Non-metric MDS (Kruskal stress)."
