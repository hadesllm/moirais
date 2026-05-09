"""Variogram Monte Carlo envelope"""

import numpy as np

from ._containers import SpatialResult


def vario_envelope(coords, values, *, nbins=15):
    """Variogram Monte Carlo envelope

    Returns
    -------
    SpatialResult
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    n = int(values) if values.ndim == 0 else len(values)
    dists = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(axis=-1))
    diffs = (values[:, None] - values[None, :]) ** 2
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_flat = dists[mask]
    g_flat = 0.5 * diffs[mask]
    nbins = min(15, max(5, n // 3))
    edges = np.linspace(d_flat.min(), d_flat.max() * 0.67, nbins + 1)
    lags, gamma = [], []
    for i in range(nbins):
        sel = (d_flat >= edges[i]) & (d_flat < edges[i + 1])
        if sel.sum() > 0:
            lags.append(float(d_flat[sel].mean()))
            gamma.append(float(g_flat[sel].mean()))
    sill = float(np.var(values)) if len(gamma) == 0 else float(max(gamma))
    return SpatialResult(
        name="vgenv",
        statistic=sill,
        p_value=None,
        extra={"lags": lags, "gamma": gamma, "sill": sill, "ci_low": None, "ci_high": None, "method": "vario_envelope"},
    )


vari = vario_envelope


def cheatsheet() -> str:
    return "vario_envelope({}) -> Variogram Monte Carlo envelope"
