"""Windowed variogram cloud (Schabenberger & Gotway Ch 8)."""

import numpy as np
from scipy.spatial.distance import pdist


def wdvar(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    max_dist: float | None = None,
    n_bins: int = 15,
) -> dict:
    """
    Compute the variogram cloud and binned empirical variogram.

    The variogram cloud contains all pairwise squared differences
    plotted against distance. The binned version averages within lag
    bins, producing the classical empirical semivariogram.

    :param coords: Observation coordinates (n, 2).
    :param values: Observed values (n,).
    :param max_dist: Maximum distance to consider; defaults to half
        the maximum pairwise distance.
    :param n_bins: Number of lag bins.
    :return: dict with ``cloud_dists``, ``cloud_gamma``, ``lags``,
        ``semivariance``, ``counts``.
    :raises ValueError: If shapes are incompatible.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 8.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    n = int(values) if values.ndim == 0 else len(values)
    if coords.shape != (n, 2):
        raise ValueError(f"coords must be ({n}, 2), got {coords.shape}.")

    pair_dists = pdist(coords)
    pair_sq_diff = pdist(values.reshape(-1, 1)) ** 2 / 2.0

    if max_dist is None:
        max_dist = float(pair_dists.max() / 2.0)

    mask = pair_dists <= max_dist
    cloud_dists = pair_dists[mask]
    cloud_gamma = pair_sq_diff[mask]

    lag_edges = np.linspace(0, max_dist, n_bins + 1)
    lag_centers = 0.5 * (lag_edges[:-1] + lag_edges[1:])
    semivariance = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for k in range(n_bins):
        bin_mask = (cloud_dists > lag_edges[k]) & (cloud_dists <= lag_edges[k + 1])
        cnt = bin_mask.sum()
        counts[k] = cnt
        if cnt > 0:
            semivariance[k] = float(cloud_gamma[bin_mask].mean())

    return {
        "cloud_dists": cloud_dists,
        "cloud_gamma": cloud_gamma,
        "lags": lag_centers,
        "semivariance": semivariance,
        "counts": counts,
        "max_dist": max_dist,
        "n": n,
    }


wdvar_fn = wdvar


def cheatsheet() -> str:
    return "wdvar({}) -> Windowed variogram cloud and binned semivariogram."
