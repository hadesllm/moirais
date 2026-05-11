# morie.fn — function file (hadesllm/morie)
"""Elbow method for optimal k."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult
from .kmcls import kmeans


def elbow_method(
    data: np.ndarray,
    k_range: tuple[int, int] = (2, 10),
    seed: int = 42,
) -> DescriptiveResult:
    """Elbow method: plot inertia vs k to find optimal number of clusters.

    Parameters
    ----------
    data : ndarray (n, p)
        Data matrix.
    k_range : tuple
        (min_k, max_k) range to evaluate.
    seed : int
        Random seed.

    Returns
    -------
    DescriptiveResult
        ``value`` is the suggested optimal k (maximum second derivative).
        ``extra`` has ``k_values`` and ``inertias``.
    """
    X = np.asarray(data, dtype=np.float64)
    k_min, k_max = k_range
    ks = list(range(k_min, k_max + 1))
    inertias = []

    for k in ks:
        res = kmeans(X, k=k, seed=seed, n_init=5)
        inertias.append(res.inertia)

    inertias_arr = np.array(inertias)
    if len(inertias_arr) >= 3:
        d2 = np.diff(inertias_arr, n=2)
        optimal_idx = int(np.argmax(np.abs(d2))) + 1
        optimal_k = ks[optimal_idx]
    else:
        optimal_k = ks[0]

    return DescriptiveResult(
        name="ElbowMethod",
        value=optimal_k,
        extra={
            "k_values": ks,
            "inertias": inertias,
        },
    )


elbmk = elbow_method


def cheatsheet() -> str:
    return "elbow_method({}) -> Elbow method for optimal k."
