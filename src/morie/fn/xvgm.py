"""Cross-variogram estimation (Schabenberger & Gotway Ch 5)."""

import numpy as np
from scipy.spatial.distance import pdist, squareform


def xvgm(
    coords: np.ndarray,
    var1: np.ndarray,
    var2: np.ndarray,
    *,
    n_lags: int = 10,
    max_dist: float | None = None,
) -> dict:
    r"""
    Compute the empirical cross-variogram between two variables.

    .. math::

        \\hat{\\gamma}_{12}(h) = \\frac{1}{2|N(h)|}
        \\sum_{(i,j) \\in N(h)}
        [Z_1(s_i) - Z_1(s_j)][Z_2(s_i) - Z_2(s_j)]

    :param coords: Observation coordinates (n, 2).
    :param var1: First variable values (n,).
    :param var2: Second variable values (n,).
    :param n_lags: Number of lag bins.
    :param max_dist: Maximum distance; defaults to half max pairwise.
    :return: dict with ``lags``, ``cross_semivariance``, ``counts``.
    :raises ValueError: If shapes are incompatible.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 5.
    """
    coords = np.asarray(coords, dtype=float)
    var1 = np.asarray(var1, dtype=float)
    var2 = np.asarray(var2, dtype=float)
    n = len(var1)
    if coords.shape != (n, 2):
        raise ValueError(f"coords must be ({n}, 2), got {coords.shape}.")
    if len(var2) != n:
        raise ValueError("var2 must have same length as var1.")

    pair_dists = pdist(coords)
    if max_dist is None:
        max_dist = float(pair_dists.max() / 2.0)

    diff1 = squareform(pdist(var1.reshape(-1, 1)))
    diff2 = squareform(pdist(var2.reshape(-1, 1)))
    sign1 = var1[:, None] - var1[None, :]
    sign2 = var2[:, None] - var2[None, :]
    cross_prod = sign1 * sign2

    triu = np.triu_indices(n, k=1)
    flat_dists = squareform(pdist(coords))[triu]
    flat_cross = cross_prod[triu]

    lag_edges = np.linspace(0, max_dist, n_lags + 1)
    lag_centers = 0.5 * (lag_edges[:-1] + lag_edges[1:])
    cross_sv = np.full(n_lags, np.nan)
    counts = np.zeros(n_lags, dtype=int)

    for k in range(n_lags):
        mask = (flat_dists > lag_edges[k]) & (flat_dists <= lag_edges[k + 1])
        cnt = mask.sum()
        counts[k] = cnt
        if cnt > 0:
            cross_sv[k] = 0.5 * float(np.mean(flat_cross[mask]))

    return {
        "lags": lag_centers,
        "cross_semivariance": cross_sv,
        "counts": counts,
        "max_dist": max_dist,
        "n": n,
    }


xvgm_fn = xvgm


def cheatsheet() -> str:
    return "xvgm({}) -> Cross-variogram estimation."
