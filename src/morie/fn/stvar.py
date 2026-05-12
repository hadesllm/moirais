"""Empirical spatiotemporal variogram gamma(h, u)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["spatiotemporal_variogram"]


def spatiotemporal_variogram(x, coords, times,
                             n_spatial_bins: int = 6,
                             n_temporal_bins: int = 6,
                             max_spatial: float | None = None,
                             max_temporal: float | None = None):
    """
    Empirical spatiotemporal semivariogram

        gamma_hat(h, u) = 1/(2 |N(h, u)|)
                         sum_{(i,j) in N(h,u)} (Z(s_i,t_i) - Z(s_j,t_j))^2.

    Parameters
    ----------
    x : array-like, shape (n,)
    coords : array-like, shape (n, d)
    times  : array-like, shape (n,)
    n_spatial_bins, n_temporal_bins : int
    max_spatial, max_temporal : float, optional

    Returns
    -------
    RichResult with payload:
        estimate : dict {gamma, spatial_bins, temporal_bins, counts}
        n, method
    """
    x = np.asarray(x, dtype=float).ravel()
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    t = np.asarray(times, dtype=float).ravel()
    n = x.size
    if coords.shape[0] != n or t.size != n:
        raise ValueError("shape mismatch among x, coords, times")
    dx = coords[:, None, :] - coords[None, :, :]
    sd = np.sqrt((dx ** 2).sum(axis=2))
    td = np.abs(t[:, None] - t[None, :])
    iu = np.triu_indices(n, k=1)
    sd_f = sd[iu]; td_f = td[iu]
    diffs2 = (x[iu[0]] - x[iu[1]]) ** 2
    if max_spatial is None:
        max_spatial = float(sd_f.max() / 2.0) if sd_f.max() > 0 else 1.0
    if max_temporal is None:
        max_temporal = float(td_f.max() / 2.0) if td_f.max() > 0 else 1.0
    s_edges = np.linspace(0.0, max_spatial, n_spatial_bins + 1)
    t_edges = np.linspace(0.0, max_temporal, n_temporal_bins + 1)
    gamma = np.full((n_spatial_bins, n_temporal_bins), np.nan)
    counts = np.zeros_like(gamma, dtype=int)
    for i in range(n_spatial_bins):
        for j in range(n_temporal_bins):
            m = (sd_f > s_edges[i]) & (sd_f <= s_edges[i + 1]) & \
                (td_f > t_edges[j]) & (td_f <= t_edges[j + 1])
            k = int(m.sum())
            counts[i, j] = k
            if k > 0:
                gamma[i, j] = 0.5 * diffs2[m].mean()
    return RichResult(payload={
        "estimate": {
            "gamma": gamma.tolist(),
            "spatial_bins": (0.5 * (s_edges[:-1] + s_edges[1:])).tolist(),
            "temporal_bins": (0.5 * (t_edges[:-1] + t_edges[1:])).tolist(),
            "counts": counts.tolist(),
        },
        "n": int(n),
        "method": "Empirical spatiotemporal variogram",
    })


def cheatsheet():
    return "stvar: Spatiotemporal variogram gamma(h,u)"


# CANONICAL TEST
# x = [i + t for (i,t) in pairs of (range(3), range(3))]
# coords[:,0] = i; times = t  -> gamma increases with h and u.
