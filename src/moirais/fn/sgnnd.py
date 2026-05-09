"""Nearest-neighbor distances for point patterns."""

from __future__ import annotations

from ._containers import DescriptiveResult


def nearest_neighbor_distances(points):
    """Compute nearest-neighbor distances for each point.

    .. epigraph:: "Bear! Bear! Run you stupid piece of..." -- Geralt, The Witcher

    Parameters
    ----------
    points : array_like
        Point coordinates, shape ``(n, 2)``.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np
    from scipy.spatial.distance import pdist, squareform

    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]

    D = squareform(pdist(pts))
    np.fill_diagonal(D, np.inf)
    nn_dists = D.min(axis=1)
    nn_indices = D.argmin(axis=1)

    return DescriptiveResult(
        name="nearest_neighbor_distances",
        value=float(nn_dists.mean()),
        extra={
            "nn_distances": nn_dists.tolist(),
            "nn_indices": nn_indices.tolist(),
            "mean_nn": float(nn_dists.mean()),
            "median_nn": float(np.median(nn_dists)),
            "std_nn": float(nn_dists.std()),
            "min_nn": float(nn_dists.min()),
            "max_nn": float(nn_dists.max()),
        },
    )


sgnnd = nearest_neighbor_distances


def cheatsheet() -> str:
    return "nearest_neighbor_distances({}) -> Nearest-neighbor distances for point patterns."
