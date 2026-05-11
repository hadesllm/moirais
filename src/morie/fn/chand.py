# morie.fn — function file (hadesllm/morie)
"""Calinski-Harabasz index."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def calinski_harabasz(
    data: np.ndarray,
    labels: np.ndarray,
) -> DescriptiveResult:
    """Calinski-Harabasz index (variance ratio criterion).

    Higher values indicate better-defined clusters.

    Parameters
    ----------
    data : ndarray (n, p)
        Data matrix.
    labels : ndarray (n,)
        Cluster labels.

    Returns
    -------
    DescriptiveResult
        ``value`` is the CH index.
    """
    X = np.asarray(data, dtype=np.float64)
    y = np.asarray(labels)
    n, p = X.shape
    classes = np.unique(y)
    k = len(classes)

    overall_mean = X.mean(axis=0)
    Bgss = 0.0
    Wgss = 0.0

    for c in classes:
        Xc = X[y == c]
        nc = Xc.shape[0]
        mc = Xc.mean(axis=0)
        Bgss += nc * np.sum((mc - overall_mean) ** 2)
        Wgss += np.sum((Xc - mc) ** 2)

    if Wgss == 0 or k <= 1 or n <= k:
        ch = 0.0
    else:
        ch = (Bgss / (k - 1)) / (Wgss / (n - k))

    return DescriptiveResult(
        name="CalinskiHarabasz",
        value=float(ch),
    )


chand = calinski_harabasz


def cheatsheet() -> str:
    return "calinski_harabasz({}) -> Calinski-Harabasz variance ratio index."
