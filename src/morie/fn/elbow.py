# morie.fn -- function file (hadesllm/morie)
"""Elbow method for optimal k in k-means clustering."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def elbow_method(
    X: Union[np.ndarray, Any],
    *,
    max_k: int = 10,
    n_init: int = 3,
    max_iter: int = 100,
    random_state: int = 42,
) -> dict[str, Any]:
    """Elbow method: run k-means for k=1..max_k and find the knee point.

    Pure NumPy k-means implementation.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Feature matrix.
    max_k : int
        Maximum number of clusters to try (default 10).
    n_init : int
        Number of random initialisations per k (default 3).
    max_iter : int
        Maximum k-means iterations (default 100).
    random_state : int
        Random seed.

    Returns
    -------
    dict
        k_values (list), inertias (list), optimal_k (int).

    References
    ----------
    Thorndike, R. L. (1953). Who belongs in the family? *Psychometrika*,
        18(4), 267-276.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    max_k = min(max_k, n)
    rng = np.random.default_rng(random_state)

    k_values = list(range(1, max_k + 1))
    inertias = []

    for k in k_values:
        best_inertia = float("inf")
        for _ in range(n_init):
            centers = X[rng.choice(n, size=k, replace=False)]
            for _ in range(max_iter):
                dists = np.array([np.sum((X - c) ** 2, axis=1) for c in centers]).T
                labels = np.argmin(dists, axis=1)
                new_centers = np.array(
                    [X[labels == j].mean(axis=0) if np.any(labels == j) else centers[j] for j in range(k)]
                )
                if np.allclose(centers, new_centers):
                    break
                centers = new_centers
            inertia = sum(np.sum((X[labels == j] - centers[j]) ** 2) for j in range(k) if np.any(labels == j))
            best_inertia = min(best_inertia, inertia)
        inertias.append(float(best_inertia))

    # Knee point detection via maximum distance from line k=1..max_k
    optimal_k = _find_knee(k_values, inertias)

    return {
        "k_values": k_values,
        "inertias": inertias,
        "optimal_k": optimal_k,
    }


def _find_knee(k_values: list[int], inertias: list[float]) -> int:
    """Find knee point via maximum perpendicular distance from the line."""
    if len(k_values) <= 2:
        return k_values[0]
    p1 = np.array([k_values[0], inertias[0]])
    p2 = np.array([k_values[-1], inertias[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-12:
        return k_values[0]

    best_dist = -1.0
    best_k = k_values[0]
    for i, (k, inert) in enumerate(zip(k_values, inertias)):
        pt = np.array([k, inert])
        # Perpendicular distance from point to line (2D cross product)
        v = p1 - pt
        dist = abs(float(line_vec[0] * v[1] - line_vec[1] * v[0])) / line_len
        if dist > best_dist:
            best_dist = dist
            best_k = k
    return best_k


elbow = elbow_method


def cheatsheet() -> str:
    return "elbow_method({}) -> Elbow method for optimal k in k-means clustering."
