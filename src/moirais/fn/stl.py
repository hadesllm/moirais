"""STL decomposition (seasonal, trend, residual)."""

import numpy as np

from ._containers import DescriptiveResult


def stl_decompose(y: np.ndarray, period: int = 12, n_iter: int = 2) -> DescriptiveResult:
    """
    STL-like decomposition into seasonal, trend, and residual components.

    Uses iterative moving-average smoothing for trend extraction and
    period-averaging for seasonal extraction.

    :param y: (n,) time series.
    :param period: Seasonal period (e.g. 12 for monthly).
    :param n_iter: Number of outer loop iterations.
    :return: DescriptiveResult with trend, seasonal, residual arrays.
    :raises ValueError: If series shorter than 2 periods.

    References
    ----------
    Cleveland RB et al. (1990). STL: A seasonal-trend decomposition
    procedure based on loess. Journal of Official Statistics, 6(1), 3-73.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(y)
    if n < 2 * period:
        raise ValueError("Series must be at least 2 full periods.")
    trend = np.zeros(n)
    seasonal = np.zeros(n)
    for _ in range(n_iter):
        detrended = y - trend
        for s in range(period):
            indices = np.arange(s, n, period)
            seasonal[indices] = detrended[indices].mean()
        seasonal -= seasonal[:period].mean()
        deseasonalised = y - seasonal
        half = period // 2
        for t in range(n):
            lo = max(0, t - half)
            hi = min(n, t + half + 1)
            trend[t] = deseasonalised[lo:hi].mean()
    residual = y - trend - seasonal
    return DescriptiveResult(
        name="stl_decompose",
        value=float(np.var(residual)),
        extra={"trend": trend, "seasonal": seasonal, "residual": residual, "period": period, "n": n},
    )


stl = stl_decompose


def cheatsheet() -> str:
    return "stl_decompose({}) -> STL decomposition (seasonal, trend, residual)."
