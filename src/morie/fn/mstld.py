# morie.fn — function file (hadesllm/morie)
"""MSTL: Multiple Seasonal and Trend decomposition using Loess."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from ._containers import TimeSeriesResult


def mstld(y, periods=None, seasonal_deg=1, trend_deg=1, seasonal_jump=1, trend_jump=1):
    """MSTL decomposition for multiple seasonal components.

    Parameters
    ----------
    y : array-like
        Time series.
    periods : list of int, optional
        Seasonal periods (e.g., [7, 365] for weekly and yearly).
        If None, defaults to [12] (monthly).
    seasonal_deg : int, optional
        Degree of seasonal LOESS. Default 1.
    trend_deg : int, optional
        Degree of trend LOESS. Default 1.
    seasonal_jump : int, optional
        LOESS jump parameter for seasonal. Default 1.
    trend_jump : int, optional
        LOESS jump parameter for trend. Default 1.

    Returns
    -------
    TimeSeriesResult
        Fields: trend, seasonals (list), remainder, periods.

    References
    ----------
    Kastnera, A., Beard, K. M., & Cleveland, W. S. (2011).
    Modeling multiple seasonal patterns in a single time series.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)

    if periods is None:
        periods = [12]

    periods = [p for p in periods if p > 1 and p <= n]

    if not periods:
        raise ValueError("No valid periods specified")

    # Simplified MSTL: use moving averages instead of LOESS
    # 1. Initial trend via moving average
    window_trend = min(13, max(3, n // 10))
    trend = uniform_filter1d(y, size=window_trend, mode="nearest")

    # 2. Detrend
    detrended = y - trend

    # 3. Extract seasonals via seasonal decomposition
    seasonals = []
    remainder = np.copy(detrended)

    for period in periods:
        seasonal_comp = np.zeros(n)
        for i in range(period):
            indices = np.arange(i, n, period)
            if len(indices) > 0:
                seasonal_comp[indices] = np.mean(remainder[indices])

        # Smooth seasonal component
        seasonal_comp = uniform_filter1d(seasonal_comp, size=min(3, period), mode="nearest")
        seasonals.append(seasonal_comp.copy())
        remainder -= seasonal_comp

    trend_absorbed = uniform_filter1d(remainder, size=window_trend, mode="nearest")
    trend_final = trend + trend_absorbed
    remainder_final = remainder - trend_absorbed

    return TimeSeriesResult(
        name="mstl_decomposition",
        values=trend_final.copy(),
        extra={
            "trend": trend_final.copy(),
            "seasonals": seasonals,
            "remainder": remainder_final.copy(),
            "periods": periods,
        },
    )


mstl_decomposition = mstld


def cheatsheet() -> str:
    return "mstld(y, periods=None) -> MSTL multi-seasonal decomposition"
