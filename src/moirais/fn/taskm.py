"""Dynamic time warping pattern matching. 'I can do anything you can do.' -- Taskmaster"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def dtw_match(
    query: np.ndarray | list[float],
    template: np.ndarray | list[float],
    *,
    window: int | None = None,
) -> DescriptiveResult:
    """Compute the dynamic time warping (DTW) distance and alignment path.

    Parameters
    ----------
    query : array-like
        Query time series (1D).
    template : array-like
        Template time series (1D).
    window : int or None
        Sakoe-Chiba band width. None = no constraint.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``distance``, ``normalized_distance``,
        ``path`` (list of (i, j) tuples), ``path_length``.
    """
    q = np.asarray(query, dtype=float)
    t = np.asarray(template, dtype=float)
    if q.ndim != 1 or t.ndim != 1:
        raise ValueError("Both inputs must be 1D")
    if len(q) == 0 or len(t) == 0:
        raise ValueError("Both inputs must be non-empty")

    n, m = len(q), len(t)
    DTW = np.full((n + 1, m + 1), np.inf)
    DTW[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - window) if window else 1
        j_end = min(m, i + window) if window else m
        for j in range(j_start, j_end + 1):
            cost = (q[i - 1] - t[j - 1]) ** 2
            DTW[i, j] = cost + min(DTW[i - 1, j], DTW[i, j - 1], DTW[i - 1, j - 1])

    dist = float(np.sqrt(DTW[n, m]))

    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = [(DTW[i - 1, j - 1], i - 1, j - 1), (DTW[i - 1, j], i - 1, j), (DTW[i, j - 1], i, j - 1)]
        _, i, j = min(candidates, key=lambda x: x[0])
    path.reverse()

    norm_dist = dist / len(path) if path else dist

    return DescriptiveResult(
        name="dtw_match",
        value={
            "distance": dist,
            "normalized_distance": float(norm_dist),
            "path": path,
            "path_length": len(path),
        },
        extra={"n_query": n, "n_template": m},
    )


taskm = dtw_match


def cheatsheet() -> str:
    return "dtw_match({}) -> Dynamic time warping pattern matching. 'I can do anything yo"
