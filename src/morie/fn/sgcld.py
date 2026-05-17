"""Variogram cloud (all pairwise differences)."""

from __future__ import annotations

from ._containers import DescriptiveResult


def variogram_cloud(Z, coords):
    """Compute the variogram cloud: all pairwise (distance, squared-difference).

    .. epigraph:: He who has a why to live can bear almost any how. -- Friedrich Nietzsche

    Parameters
    ----------
    Z : array_like
        Observed values.
    coords : array_like
        Coordinates, shape ``(n, 2)``.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np
    from scipy.spatial.distance import pdist, squareform

    Z = np.asarray(Z, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)

    D = squareform(pdist(coords))
    n = len(Z)
    idx = np.triu_indices(n, k=1)

    distances = D[idx]
    sq_diffs = 0.5 * (Z[idx[0]] - Z[idx[1]]) ** 2

    return DescriptiveResult(
        name="variogram_cloud",
        value=float(sq_diffs.mean()),
        extra={
            "distances": distances,
            "squared_differences": sq_diffs,
            "n_pairs": len(distances),
            "mean_sqdiff": float(sq_diffs.mean()),
        },
    )


sgcld = variogram_cloud


def cheatsheet() -> str:
    return "variogram_cloud({}) -> Variogram cloud (all pairwise differences)."
