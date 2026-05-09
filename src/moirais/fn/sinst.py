"""Volatility of volatility (VIX-like). 'In blackest night, you will fear me.' -- Sinestro"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult
from ._helpers import _extract_col


def vol_of_vol(
    data: pd.DataFrame | np.ndarray,
    *,
    col: str = "x",
    window: int = 20,
) -> DescriptiveResult:
    """Volatility of volatility: rolling standard deviation of rolling volatility.

    First computes a rolling standard deviation (realized volatility), then
    computes the standard deviation of that volatility series.  Analogous to
    the VIX (fear index) concept in finance.

    Parameters
    ----------
    data : DataFrame or array
        Input time series (e.g. returns or prices).
    col : str
        Column name if *data* is a DataFrame.
    window : int
        Rolling window size.

    Returns
    -------
    DescriptiveResult
        ``value`` = vol-of-vol (scalar).
    """
    x = _extract_col(data, col)
    n = len(x)
    if n < 2 * window:
        raise ValueError(f"Need at least {2 * window} observations (got {n})")
    rolling_vol = np.array([np.std(x[i : i + window], ddof=1) for i in range(n - window + 1)])
    vov = float(np.std(rolling_vol, ddof=1))
    return DescriptiveResult(
        name="Volatility of volatility",
        value=vov,
        extra={
            "n": n,
            "window": window,
            "mean_vol": float(np.mean(rolling_vol)),
            "max_vol": float(np.max(rolling_vol)),
            "min_vol": float(np.min(rolling_vol)),
            "vol_series_length": len(rolling_vol),
            "fear_ratio": float(vov / np.mean(rolling_vol)) if np.mean(rolling_vol) > 0 else 0.0,
        },
    )


sinst = vol_of_vol


def cheatsheet() -> str:
    return "vol_of_vol({}) -> Volatility of volatility (VIX-like). 'In blackest night, you"
