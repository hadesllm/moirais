"""Empirical spatiotemporal autocovariance C(h, u)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["spatiotemporal_autocovariance"]


def spatiotemporal_autocovariance(x, coords, times,
                                  n_spatial_bins: int = 6,
                                  n_temporal_bins: int = 6,
                                  max_spatial: float | None = None,
                                  max_temporal: float | None = None):
    """
    Empirical spatiotemporal autocovariance.

    Estimator (centred sample covariance per (spatial-lag, temporal-lag)
    bin):
        C_hat(h, u) = 1/|N(h,u)| sum_{(i,j) in N(h,u)}
                      (Z(s_i,t_i) - mean)(Z(s_j,t_j) - mean).

    Parameters
    ----------
    x : array-like, shape (n,)
    coords : array-like, shape (n, d)
    times  : array-like, shape (n,)
    n_spatial_bins, n_temporal_bins : int
    max_spatial, max_temporal : float, optional
        Upper distance cutoffs. Default = max(distance)/2.

    Returns
    -------
    RichResult with payload:
        estimate : dict {C : (n_s x n_t) covariance matrix,
                         spatial_bins, temporal_bins, counts}
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
    xbar = x.mean()
    dx = coords[:, None, :] - coords[None, :, :]
    sd = np.sqrt((dx ** 2).sum(axis=2))
    td = np.abs(t[:, None] - t[None, :])
    iu = np.triu_indices(n, k=1)
    sd_f = sd[iu]; td_f = td[iu]
    prods = (x[iu[0]] - xbar) * (x[iu[1]] - xbar)
    if max_spatial is None:
        max_spatial = float(sd_f.max() / 2.0) if sd_f.max() > 0 else 1.0
    if max_temporal is None:
        max_temporal = float(td_f.max() / 2.0) if td_f.max() > 0 else 1.0
    s_edges = np.linspace(0.0, max_spatial, n_spatial_bins + 1)
    t_edges = np.linspace(0.0, max_temporal, n_temporal_bins + 1)
    C = np.full((n_spatial_bins, n_temporal_bins), np.nan)
    counts = np.zeros_like(C, dtype=int)
    for i in range(n_spatial_bins):
        for j in range(n_temporal_bins):
            m = (sd_f > s_edges[i]) & (sd_f <= s_edges[i + 1]) & \
                (td_f > t_edges[j]) & (td_f <= t_edges[j + 1])
            k = int(m.sum())
            counts[i, j] = k
            if k > 0:
                C[i, j] = float(prods[m].mean())
    return RichResult(payload={
        "estimate": {
            "C": C.tolist(),
            "spatial_bins": (0.5 * (s_edges[:-1] + s_edges[1:])).tolist(),
            "temporal_bins": (0.5 * (t_edges[:-1] + t_edges[1:])).tolist(),
            "counts": counts.tolist(),
        },
        "n": int(n),
        "method": "Empirical spatiotemporal autocovariance",
    })


def cheatsheet():
    return "stacv: Spatiotemporal autocovariance C(h,u)"


# CANONICAL TEST
# x = i + t for (i,t) in product(range(3), range(3))  -> n=9
# coords = i column, times = t column
# Bins are populated; covariance increases with both h and u.
