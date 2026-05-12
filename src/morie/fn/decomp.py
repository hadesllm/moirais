# morie.fn -- function file (hadesllm/morie)
"""We suffer more often in imagination than in reality. -- Seneca"""

from __future__ import annotations

import numpy as np

from ._containers import TimeSeriesResult


def seasonal_decompose(x, *, period: int = 12, model: str = "additive") -> TimeSeriesResult:
    """Classical seasonal decomposition (moving average).

    Parameters
    ----------
    x : array-like
        Time series observations.
    period : int
        Seasonal period. Default 12.
    model : str
        ``"additive"`` or ``"multiplicative"``. Default ``"additive"``.

    Returns
    -------
    TimeSeriesResult
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if n < 2 * period:
        raise ValueError(f"Need at least {2 * period} observations for period={period}")
    # Trend: centered moving average
    trend = np.full(n, np.nan)
    half = period // 2
    for i in range(half, n - half):
        if period % 2 == 0:
            trend[i] = (x[i - half] / 2 + np.sum(x[i - half + 1 : i + half]) + x[i + half] / 2) / period
        else:
            trend[i] = np.mean(x[i - half : i + half + 1])
    # Seasonal component
    if model == "additive":
        detrended = x - trend
    else:
        detrended = x / np.where(trend > 0, trend, np.nan)
    seasonal = np.zeros(n)
    for s in range(period):
        idx = np.arange(s, n, period)
        vals = detrended[idx]
        seasonal[idx] = float(np.nanmean(vals))
    # Remainder
    if model == "additive":
        remainder = x - trend - seasonal
    else:
        remainder = x / (np.where(trend > 0, trend, 1) * np.where(seasonal > 0, seasonal, 1))
    return TimeSeriesResult(
        name=f"Decomposition ({model})",
        values=trend,
        extra={
            "trend": trend.tolist(),
            "seasonal": seasonal.tolist(),
            "remainder": remainder.tolist(),
            "period": period,
            "model": model,
        },
    )


decomp = seasonal_decompose


def cheatsheet() -> str:
    return "seasonal_decompose({}) -> Seasonal decomposition. 'The circle is now complete.' -- Dar"
