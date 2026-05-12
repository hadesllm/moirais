# morie.fn -- function file (hadesllm/morie)
"""K-means clustering via Lloyd's algorithm with k-means++ initialisation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._containers import KmeansRes


def _kmeans_pp(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """K-means++ initialisation (Arthur & Vassilvitskii, 2007)."""
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=np.float64)
    centers[0] = X[rng.integers(0, n)]

    for c in range(1, k):
        dists = np.min(
            np.sum((X[:, None, :] - centers[None, :c, :]) ** 2, axis=2),
            axis=1,
        )
        probs = dists / dists.sum()
        centers[c] = X[rng.choice(n, p=probs)]
    return centers


def kmean(
    data: pd.DataFrame | np.ndarray,
    k: int = 3,
    n_init: int = 10,
    max_iter: int = 300,
    seed: int | None = 42,
) -> KmeansRes:
    """K-means clustering.

    Runs Lloyd's algorithm *n_init* times with k-means++ initialisation
    and returns the result with lowest inertia (within-cluster sum of
    squared distances).

    Parameters
    ----------
    data : DataFrame or ndarray
        Observations as rows.
    k : int
        Number of clusters.
    n_init : int
        Number of independent initialisations.
    max_iter : int
        Maximum iterations per run.
    seed : int, optional
        Random seed.

    Returns
    -------
    KmeansRes
        ``labels``, ``centers``, ``inertia``, ``n_iter``.

    References
    ----------
    Arthur, D. & Vassilvitskii, S. (2007). k-means++: The advantages of
    careful seeding. *Proc. ACM-SIAM SODA*, 1027-1035.
    """
    X = np.asarray(data, dtype=np.float64)
    n = X.shape[0]
    rng = np.random.default_rng(seed)

    best_inertia = np.inf
    best_labels = np.zeros(n, dtype=int)
    best_centers = np.zeros((k, X.shape[1]))
    best_n_iter = 0

    for _ in range(n_init):
        centers = _kmeans_pp(X, k, rng)
        labels = np.zeros(n, dtype=int)

        for it in range(1, max_iter + 1):
            # Assignment
            dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels_new = np.argmin(dists, axis=1)

            # Update
            new_centers = np.empty_like(centers)
            for c in range(k):
                members = X[labels_new == c]
                if len(members) > 0:
                    new_centers[c] = members.mean(axis=0)
                else:
                    new_centers[c] = X[rng.integers(0, n)]

            if np.allclose(centers, new_centers):
                centers = new_centers
                labels = labels_new
                n_iter_run = it
                break
            centers = new_centers
            labels = labels_new
            n_iter_run = it

        inertia = sum(np.sum((X[labels == c] - centers[c]) ** 2) for c in range(k))

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centers = centers
            best_n_iter = n_iter_run

    return KmeansRes(
        labels=best_labels,
        centers=best_centers,
        inertia=float(best_inertia),
        n_iter=best_n_iter,
    )


def cheatsheet() -> str:
    return "_kmeans_pp({}) -> K-means clustering via Lloyd's algorithm with k-means++ init"
