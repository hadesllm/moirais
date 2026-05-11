"""Lag class binning for variogram estimation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def lag_class_binning(distances, n_bins=15, max_dist=None):
    """Bin pairwise distances into lag classes for variogram estimation.

    .. epigraph:: "You Died." -- Dark Souls

    Parameters
    ----------
    distances : array_like
        Pairwise distances (upper triangle).
    n_bins : int
        Number of lag bins.
    max_dist : float, optional
        Maximum distance to consider.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np

    d = np.asarray(distances, dtype=np.float64).ravel()
    d = d[d > 0]

    if max_dist is None:
        max_dist = d.max() / 2.0

    edges = np.linspace(0, max_dist, n_bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    counts = np.zeros(n_bins, dtype=int)
    mean_dists = np.zeros(n_bins)

    for k in range(n_bins):
        mask = (d > edges[k]) & (d <= edges[k + 1])
        counts[k] = int(mask.sum())
        if counts[k] > 0:
            mean_dists[k] = d[mask].mean()
        else:
            mean_dists[k] = mids[k]

    return DescriptiveResult(
        name="lag_class_binning",
        value=float(n_bins),
        extra={
            "bin_edges": edges.tolist(),
            "bin_midpoints": mids.tolist(),
            "bin_counts": counts.tolist(),
            "mean_distances": mean_dists.tolist(),
            "bin_width": float(edges[1] - edges[0]),
        },
    )


sglgc = lag_class_binning


def cheatsheet() -> str:
    return "lag_class_binning({}) -> Lag class binning for variogram estimation."
