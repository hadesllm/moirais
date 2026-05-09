"""Silhouette score."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from ._containers import DescriptiveResult


def silhouette(
    data: np.ndarray,
    labels: np.ndarray,
) -> DescriptiveResult:
    """Silhouette coefficient for cluster quality evaluation.

    Parameters
    ----------
    data : ndarray (n, p)
        Data matrix.
    labels : ndarray (n,)
        Cluster labels.

    Returns
    -------
    DescriptiveResult
        ``value`` is the mean silhouette score.
        ``extra`` has ``per_sample`` silhouette values.
    """
    X = np.asarray(data, dtype=np.float64)
    y = np.asarray(labels)
    n = X.shape[0]
    D = squareform(pdist(X))

    clusters = np.unique(y)
    sil = np.zeros(n)

    for i in range(n):
        ci = y[i]
        same = y == ci
        same[i] = False

        if np.sum(same) == 0:
            sil[i] = 0.0
            continue

        a_i = np.mean(D[i, same])

        b_i = np.inf
        for c in clusters:
            if c == ci:
                continue
            other = y == c
            if np.any(other):
                b_i = min(b_i, np.mean(D[i, other]))

        denom = max(a_i, b_i)
        sil[i] = (b_i - a_i) / denom if denom > 0 else 0.0

    return DescriptiveResult(
        name="Silhouette",
        value=float(np.mean(sil)),
        extra={"per_sample": sil},
    )


sihlh = silhouette


def cheatsheet() -> str:
    return "silhouette({}) -> Silhouette coefficient for clustering."
