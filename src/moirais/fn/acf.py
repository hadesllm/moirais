# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Autocorrelation function."""

from __future__ import annotations

import numpy as np

from ._containers import TimeSeriesResult


def autocorrelation(x, *, max_lag: int = 20) -> TimeSeriesResult:
    """Sample autocorrelation function (ACF) up to max_lag.

    Parameters
    ----------
    x : array-like
        Time series observations.
    max_lag : int
        Maximum lag to compute. Default 20.

    Returns
    -------
    TimeSeriesResult
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 observations")
    x_centered = x - x.mean()
    var = np.sum(x_centered**2) / n
    max_lag = min(max_lag, n - 1)
    acf_vals = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        acf_vals[k] = np.sum(x_centered[: n - k] * x_centered[k:]) / (n * var) if var > 0 else 0
    lags = np.arange(max_lag + 1)
    ci = 1.96 / np.sqrt(n)
    return TimeSeriesResult(
        name="ACF",
        values=acf_vals,
        lags=lags,
        ci_upper=np.full(max_lag + 1, ci),
        ci_lower=np.full(max_lag + 1, -ci),
        extra={"n": n, "ci_bound": float(ci)},
    )


acf = autocorrelation


def cheatsheet() -> str:
    return "autocorrelation({}) -> Autocorrelation function."
