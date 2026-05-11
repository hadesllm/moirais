"""Effective range estimation"""

import numpy as np

from ._containers import SpatialResult


def range_est(data, *, method="default"):
    """Effective range estimation

    Returns
    -------
    SpatialResult
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        values = data.copy()
        coords = np.arange(len(values), dtype=float).reshape(-1, 1)
    else:
        values = data[:, 0].copy()
        coords = data
    n = len(values)
    dists = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(axis=-1))
    diffs = (values[:, None] - values[None, :]) ** 2
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_flat = dists[mask]
    g_flat = 0.5 * diffs[mask]
    n_pairs = len(d_flat)
    nbins = min(15, max(2, min(n_pairs, n // 3)))
    edges = np.linspace(d_flat.min(), d_flat.max() * 0.67, nbins + 1)
    lags, gamma = [], []
    for i in range(nbins):
        sel = (d_flat >= edges[i]) & (d_flat < edges[i + 1])
        if sel.sum() > 0:
            lags.append(float(d_flat[sel].mean()))
            gamma.append(float(g_flat[sel].mean()))
    sill = float(np.var(values)) if len(gamma) == 0 else float(max(gamma))
    return SpatialResult(
        name="vgrng",
        statistic=sill,
        p_value=None,
        extra={"lags": lags, "gamma": gamma, "sill": sill, "ci_low": None, "ci_high": None, "method": "range_est"},
    )


rang = range_est


def cheatsheet() -> str:
    return "range_est({}) -> Effective range estimation"
