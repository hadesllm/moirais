# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Naive time series forecasting. 'I am from the future.' -- Booster Gold"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult
from ._helpers import _extract_col


def naive_forecast(
    data: pd.DataFrame | np.ndarray,
    *,
    col: str = "x",
    h: int = 10,
    method: str = "naive",
    seasonal_period: int | None = None,
) -> DescriptiveResult:
    """Naive time series forecasting with prediction intervals.

    Parameters
    ----------
    data : DataFrame or array
        Historical time series.
    col : str
        Column name if *data* is a DataFrame.
    h : int
        Forecast horizon.
    method : str
        ``'naive'`` (last value), ``'drift'`` (last value + trend),
        ``'mean'`` (historical mean), or ``'snaive'`` (seasonal naive).
    seasonal_period : int or None
        Period for seasonal naive (required if method='snaive').

    Returns
    -------
    DescriptiveResult
        ``value`` = dict with ``'forecast'``, ``'lower'``, ``'upper'``.
    """
    x = _extract_col(data, col)
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 observations")
    if h < 1:
        raise ValueError("h must be >= 1")
    sigma = float(np.std(np.diff(x), ddof=1))
    if method == "naive":
        fc = np.full(h, x[-1])
        se = sigma * np.sqrt(np.arange(1, h + 1))
    elif method == "drift":
        slope = (x[-1] - x[0]) / (n - 1)
        fc = x[-1] + slope * np.arange(1, h + 1)
        se = sigma * np.sqrt(np.arange(1, h + 1) * (1 + np.arange(1, h + 1) / n))
    elif method == "mean":
        fc = np.full(h, np.mean(x))
        se = np.full(h, float(np.std(x, ddof=1)) * np.sqrt(1 + 1 / n))
    elif method == "snaive":
        if seasonal_period is None:
            raise ValueError("seasonal_period required for snaive")
        if seasonal_period < 2:
            raise ValueError("seasonal_period must be >= 2")
        fc = np.array([x[-(seasonal_period - i % seasonal_period)] for i in range(h)])
        k = np.arange(1, h + 1) // seasonal_period + 1
        se = sigma * np.sqrt(k.astype(float))
    else:
        raise ValueError(f"Unknown method: {method}")
    lower = fc - 1.96 * se
    upper = fc + 1.96 * se
    return DescriptiveResult(
        name=f"Naive forecast ({method})",
        value={"forecast": fc.tolist(), "lower": lower.tolist(), "upper": upper.tolist()},
        extra={
            "method": method,
            "h": h,
            "n": n,
            "sigma": sigma,
            "last_value": float(x[-1]),
            "mean_value": float(np.mean(x)),
        },
    )


bostr = naive_forecast


def cheatsheet() -> str:
    return "naive_forecast({}) -> Naive time series forecasting. 'I am from the future.' -- Bo"
