"""Silhouette score for cluster quality evaluation."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def silhouette_score(
    X: Union[np.ndarray, Any],
    labels: Union[np.ndarray, Any],
) -> dict[str, Any]:
    """Compute silhouette score for a clustering.

    s(i) = (b(i) - a(i)) / max(a(i), b(i))

    where a(i) is the mean intra-cluster distance and b(i) is the mean
    nearest-cluster distance.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Feature matrix.
    labels : array-like of shape (n,)
        Cluster labels.

    Returns
    -------
    dict
        mean_score (float), per_sample_scores (n,).

    Raises
    ------
    ValueError
        If fewer than 2 clusters are present.

    References
    ----------
    Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the
        interpretation and validation of cluster analysis. *Journal of
        Computational and Applied Mathematics*, 20, 53-65.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels).ravel()
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels must have same number of rows.")

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("Silhouette score requires at least 2 clusters.")

    n = X.shape[0]
    scores = np.zeros(n)

    for i in range(n):
        own_label = labels[i]
        own_mask = labels == own_label
        own_count = own_mask.sum()

        # a(i): mean distance to same-cluster points
        if own_count > 1:
            dists_own = np.sqrt(np.sum((X[own_mask] - X[i]) ** 2, axis=1))
            a_i = dists_own.sum() / (own_count - 1)
        else:
            a_i = 0.0

        # b(i): minimum mean distance to any other cluster
        b_i = float("inf")
        for lbl in unique_labels:
            if lbl == own_label:
                continue
            other_mask = labels == lbl
            dists_other = np.sqrt(np.sum((X[other_mask] - X[i]) ** 2, axis=1))
            mean_dist = dists_other.mean()
            if mean_dist < b_i:
                b_i = mean_dist

        denom = max(a_i, b_i)
        scores[i] = (b_i - a_i) / denom if denom > 0 else 0.0

    return {
        "mean_score": float(np.mean(scores)),
        "per_sample_scores": scores,
    }


silh = silhouette_score


def cheatsheet() -> str:
    return "silhouette_score({}) -> Silhouette score for cluster quality evaluation."
