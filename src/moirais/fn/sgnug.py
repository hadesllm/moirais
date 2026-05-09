"""Nugget effect estimation from empirical variogram."""

from __future__ import annotations

from ._containers import DescriptiveResult


def nugget_effect_estimate(gamma_values, lag_distances):
    """Estimate nugget, sill, and range from an empirical variogram.

    Uses linear extrapolation of the first two lag bins to estimate the
    nugget (y-intercept), and finds the sill and effective range.

    .. epigraph:: "Praise the Sun!" -- Solaire, Dark Souls

    Parameters
    ----------
    gamma_values : array_like
        Semivariance values at each lag.
    lag_distances : array_like
        Lag distances corresponding to gamma_values.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np

    gamma = np.asarray(gamma_values, dtype=np.float64)
    lags = np.asarray(lag_distances, dtype=np.float64)

    valid = gamma > 0
    if valid.sum() < 2:
        return DescriptiveResult(
            name="nugget_effect_estimate",
            value=0.0,
            extra={"nugget": 0.0, "sill": 0.0, "range": 0.0},
        )

    g = gamma[valid]
    h = lags[valid]

    slope = (g[1] - g[0]) / (h[1] - h[0]) if h[1] != h[0] else 0.0
    nugget = max(0.0, float(g[0] - slope * h[0]))
    sill = float(g.max())
    threshold = nugget + 0.95 * (sill - nugget)
    above = np.where(g >= threshold)[0]
    eff_range = float(h[above[0]]) if len(above) > 0 else float(h[-1])

    return DescriptiveResult(
        name="nugget_effect_estimate",
        value=nugget,
        extra={
            "nugget": nugget,
            "sill": sill,
            "range": eff_range,
            "nugget_sill_ratio": nugget / sill if sill > 0 else 0.0,
        },
    )


sgnug = nugget_effect_estimate


def cheatsheet() -> str:
    return "nugget_effect_estimate({}) -> Nugget effect estimation from empirical variogram."
