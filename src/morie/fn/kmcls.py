# morie.fn -- function file (hadesllm/morie)
"""K-means clustering."""

from __future__ import annotations

import numpy as np

from ._containers import KmeansRes


def kmeans(
    data: np.ndarray,
    k: int = 3,
    max_iter: int = 300,
    seed: int = 42,
    n_init: int = 10,
) -> KmeansRes:
    """K-means clustering (Lloyd's algorithm with multiple restarts).

    Parameters
    ----------
    data : ndarray (n, p)
        Data matrix.
    k : int
        Number of clusters.
    max_iter : int
        Maximum iterations per restart.
    seed : int
        Random seed.
    n_init : int
        Number of random initialisations.

    Returns
    -------
    KmeansRes
    """
    X = np.asarray(data, dtype=np.float64)
    n, p = X.shape
    rng = np.random.default_rng(seed)

    best_labels = None
    best_centers = None
    best_inertia = float("inf")
    best_iters = 0

    for _ in range(n_init):
        idx = rng.choice(n, size=k, replace=False)
        centers = X[idx].copy()

        for it in range(max_iter):
            dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)

            new_centers = np.zeros_like(centers)
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    new_centers[j] = X[mask].mean(axis=0)
                else:
                    new_centers[j] = X[rng.integers(n)]

            if np.allclose(centers, new_centers):
                centers = new_centers
                break
            centers = new_centers

        inertia = sum(np.sum((X[labels == j] - centers[j]) ** 2) for j in range(k))

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centers = centers
            best_iters = it + 1

    return KmeansRes(
        labels=best_labels,
        centers=best_centers,
        inertia=float(best_inertia),
        n_iter=best_iters,
    )


kmcls = kmeans


def cheatsheet() -> str:
    return "kmeans({}) -> K-means clustering (Lloyd's algorithm)."
