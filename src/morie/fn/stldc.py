"""STL decomposition (seasonal-trend via LOESS)."""

import numpy as np

from ._containers import DescriptiveResult


def stl_decompose(
    y: np.ndarray, period: int = 12, trend_window: int | None = None
) -> DescriptiveResult:
    """
    STL-like decomposition into trend, seasonal, and remainder.

    Uses iterative moving-average extraction (simplified STL).

    :param y: 1-D time series.
    :param period: Seasonal period. Default 12.
    :param trend_window: Window for trend smoother. Default 2*period+1.
    :return: DescriptiveResult with trend, seasonal, remainder arrays.
    :raises ValueError: If series too short.

    References
    ----------
    Cleveland R.B., Cleveland W.S., McRae J.E. & Terpenning I. (1990).
    STL: A seasonal-trend decomposition procedure based on loess.
    *Journal of Official Statistics*, 6(1), 3-73.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if n < 2 * period:
        raise ValueError(f"Need at least {2 * period} observations, got {n}.")
    if trend_window is None:
        trend_window = 2 * period + 1
    if trend_window % 2 == 0:
        trend_window += 1

    def moving_avg(x, w):
        hw = w // 2
        out = np.full(len(x), np.nan)
        for i in range(hw, len(x) - hw):
            out[i] = np.mean(x[i - hw : i + hw + 1])
        out[:hw] = out[hw]
        out[-hw:] = out[-(hw + 1)]
        return out

    trend = moving_avg(y, trend_window)
    detrended = y - trend
    seasonal = np.zeros(n)
    for s in range(period):
        indices = list(range(s, n, period))
        cycle_mean = np.mean(detrended[indices])
        for idx in indices:
            seasonal[idx] = cycle_mean
    seasonal -= np.mean(seasonal)
    remainder = y - trend - seasonal
    return DescriptiveResult(
        name="stl_decompose",
        value=float(np.std(remainder)),
        extra={
            "trend": trend,
            "seasonal": seasonal,
            "remainder": remainder,
            "period": period,
            "n": n,
        },
    )


stldc = stl_decompose


def cheatsheet() -> str:
    return "stl_decompose({}) -> STL seasonal-trend decomposition."
