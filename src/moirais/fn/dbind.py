# moirais.fn — function file (hadesllm/moirais)
"""Davies-Bouldin index."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def davies_bouldin(
    data: np.ndarray,
    labels: np.ndarray,
) -> DescriptiveResult:
    """Davies-Bouldin index for cluster evaluation.

    Lower values indicate better clustering.

    Parameters
    ----------
    data : ndarray (n, p)
        Data matrix.
    labels : ndarray (n,)
        Cluster labels.

    Returns
    -------
    DescriptiveResult
        ``value`` is the DB index.
    """
    X = np.asarray(data, dtype=np.float64)
    y = np.asarray(labels)
    classes = np.unique(y)
    k = len(classes)

    if k <= 1:
        return DescriptiveResult(name="DaviesBouldin", value=0.0)

    centers = np.zeros((k, X.shape[1]))
    scatter = np.zeros(k)

    for j, c in enumerate(classes):
        Xc = X[y == c]
        centers[j] = Xc.mean(axis=0)
        scatter[j] = np.mean(np.sqrt(np.sum((Xc - centers[j]) ** 2, axis=1)))

    db = 0.0
    for i in range(k):
        max_ratio = 0.0
        for j in range(k):
            if i == j:
                continue
            d_ij = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if d_ij > 0:
                ratio = (scatter[i] + scatter[j]) / d_ij
                max_ratio = max(max_ratio, ratio)
        db += max_ratio

    db /= k

    return DescriptiveResult(
        name="DaviesBouldin",
        value=float(db),
    )


dbind = davies_bouldin


def cheatsheet() -> str:
    return "davies_bouldin({}) -> Davies-Bouldin cluster evaluation index."
