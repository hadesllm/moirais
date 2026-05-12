"""Confine yourself to the present. -- Marcus Aurelius"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from ._containers import DescriptiveResult


def silhouette_score(
    X: np.ndarray,
    labels: np.ndarray,
) -> DescriptiveResult:
    r"""
    Compute silhouette coefficients for each sample and the mean score.

    For each sample *i* with cluster label :math:`C_i`:

    .. math::

        s(i) = \\frac{b(i) - a(i)}{\\max(a(i),\\, b(i))}

    :param X: Data matrix (n_samples, n_features).
    :param labels: Cluster assignment for each sample (integer array).
    :return: DescriptiveResult with mean silhouette as value.
    :raises ValueError: If X/labels shape mismatch or fewer than 2 clusters.

    References
    ----------
    Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the
    interpretation and validation of cluster analysis. Journal of
    Computational and Applied Mathematics, 20, 53--65.
    doi:10.1016/0377-0427(87)90125-7
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if X.ndim != 2 or labels.ndim != 1 or X.shape[0] != labels.shape[0]:
        raise ValueError("X must be (n, p) and labels must be (n,).")
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("Need at least 2 clusters for silhouette analysis.")

    n = X.shape[0]
    dist = cdist(X, X)
    sil = np.zeros(n)

    for i in range(n):
        ci = labels[i]
        mask_same = labels == ci
        mask_same[i] = False
        if np.sum(mask_same) == 0:
            sil[i] = 0.0
            continue
        a_i = np.mean(dist[i, mask_same])

        b_i = np.inf
        for cl in unique_labels:
            if cl == ci:
                continue
            mask_other = labels == cl
            b_i = min(b_i, np.mean(dist[i, mask_other]))

        sil[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0

    mean_sil = float(np.mean(sil))

    return DescriptiveResult(
        name="Silhouette Score",
        value=float(np.round(mean_sil, 4)),
        extra={
            "mean_silhouette": float(np.round(mean_sil, 4)),
            "per_sample": sil,
            "n_clusters": len(unique_labels),
            "n_samples": n,
        },
    )


silht = silhouette_score


def cheatsheet() -> str:
    return "silhouette_score({}) -> Silhouette score. 'You underestimate my power.' -- Anakin Sk"
