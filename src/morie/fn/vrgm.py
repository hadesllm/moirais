"""Empirical variogram estimation (Matheron classical estimator)."""
import numpy as np
from scipy.spatial.distance import pdist, squareform
from ._richresult import RichResult

__all__ = ["variogram_estimation"]


def variogram_estimation(x, coords, n_bins: int = 10, max_dist: float | None = None):
    """
    Empirical (Matheron) variogram.

    Formula:  gamma_hat(h) = 1/(2|N(h)|) * sum_{(i,j) in N(h)} (Z(s_i) - Z(s_j))^2

    Distances are binned into `n_bins` equal-width bins from 0 to max_dist
    (default: half the maximum inter-point distance, the standard cutoff).

    Parameters
    ----------
    x : array-like, shape (n,)
        Observed values.
    coords : array-like, shape (n, d)
        Spatial coordinates.
    n_bins : int
        Number of distance bins (default 10).
    max_dist : float, optional
        Upper distance cutoff. Default = max(pdist)/2.

    Returns
    -------
    RichResult with payload:
        estimate : dict with bins (lag midpoints), gamma (semivariance),
                   n_pairs per bin.
        n, method
    """
    x = np.asarray(x, dtype=float).ravel()
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    n = x.size
    if coords.shape[0] != n:
        raise ValueError(f"coords rows ({coords.shape[0]}) must match x ({n})")
    if n < 2:
        raise ValueError("need at least 2 points")

    d = pdist(coords)
    if max_dist is None:
        max_dist = float(d.max() / 2.0)
    # Squared differences for all pairs
    D = squareform(d)
    iu = np.triu_indices(n, k=1)
    dists = D[iu]
    diffs2 = (x[iu[0]] - x[iu[1]]) ** 2

    edges = np.linspace(0.0, max_dist, n_bins + 1)
    bins_mid = 0.5 * (edges[1:] + edges[:-1])
    gamma = np.full(n_bins, np.nan)
    npairs = np.zeros(n_bins, dtype=int)
    for k in range(n_bins):
        mask = (dists > edges[k]) & (dists <= edges[k + 1])
        m = int(mask.sum())
        npairs[k] = m
        if m > 0:
            gamma[k] = 0.5 * diffs2[mask].mean()

    estimate = {
        "bins": bins_mid.tolist(),
        "gamma": gamma.tolist(),
        "n_pairs": npairs.tolist(),
    }
    return RichResult(payload={
        "estimate": estimate,
        "n": int(n),
        "method": "Empirical (Matheron) variogram",
    })


def cheatsheet():
    return "vrgm: Empirical variogram (Matheron estimator)"


# CANONICAL TEST
# x = [1,2,3,4,5];  coords = [[0],[1],[2],[3],[4]];  n_bins=4, max_dist=4
# Pairs (h=1 distance): 4 pairs, diffs2 = [1,1,1,1] -> gamma = 0.5*mean = 0.5
# Pairs (h=2): 3 pairs, diffs2 = [4,4,4] -> gamma = 2.0
# Pairs (h=3): 2 pairs, diffs2 = [9,9]   -> gamma = 4.5
# Pairs (h=4): 1 pair,  diffs2 = [16]    -> gamma = 8.0
