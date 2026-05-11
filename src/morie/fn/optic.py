# morie.fn — function file (hadesllm/morie)
"""OPTICS clustering."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from ._containers import DescriptiveResult


def optics(
    data: np.ndarray,
    min_samples: int = 5,
    max_eps: float = float("inf"),
    xi: float = 0.05,
) -> DescriptiveResult:
    """OPTICS (Ordering Points To Identify Clustering Structure).

    Parameters
    ----------
    data : ndarray (n, p)
        Data matrix.
    min_samples : int
        Minimum neighbourhood size.
    max_eps : float
        Maximum neighbourhood radius.
    xi : float
        Steepness threshold for cluster extraction.

    Returns
    -------
    DescriptiveResult
        ``value`` is cluster labels (-1 for noise).
        ``extra`` has ``ordering``, ``reachability``, ``core_distances``.
    """
    X = np.asarray(data, dtype=np.float64)
    n = X.shape[0]
    D = squareform(pdist(X))

    core_dist = np.full(n, np.inf)
    for i in range(n):
        dists_i = np.sort(D[i])
        if len(dists_i) > min_samples:
            core_dist[i] = dists_i[min_samples]

    processed = np.zeros(n, dtype=bool)
    ordering = []
    reachability = np.full(n, np.inf)

    seeds = list(range(n))

    while len(ordering) < n:
        unprocessed = [i for i in range(n) if not processed[i]]
        if not unprocessed:
            break

        if ordering:
            candidates = [(reachability[i], i) for i in unprocessed]
            candidates.sort()
            p_idx = candidates[0][1]
        else:
            p_idx = unprocessed[0]

        processed[p_idx] = True
        ordering.append(p_idx)

        if core_dist[p_idx] <= max_eps:
            for q in range(n):
                if processed[q]:
                    continue
                new_reach = max(core_dist[p_idx], D[p_idx, q])
                if new_reach < max_eps:
                    reachability[q] = min(reachability[q], new_reach)

    labels = np.full(n, -1, dtype=int)
    cluster_id = 0
    reach_ordered = reachability[ordering]

    i = 0
    while i < n:
        if reach_ordered[i] > (1 - xi) * (np.median(reach_ordered[reach_ordered < np.inf]) if np.any(reach_ordered < np.inf) else 1.0):
            i += 1
            continue

        start = i
        while i < n and reach_ordered[i] <= max_eps:
            labels[ordering[i]] = cluster_id
            i += 1
        if i > start:
            cluster_id += 1

    return DescriptiveResult(
        name="OPTICS",
        value=labels,
        extra={
            "ordering": np.array(ordering),
            "reachability": reachability,
            "core_distances": core_dist,
            "n_clusters": cluster_id,
        },
    )


optic_fn = optics


def cheatsheet() -> str:
    return "optics({}) -> OPTICS density-based clustering."
