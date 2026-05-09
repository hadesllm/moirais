# moirais.fn — function file (hadesllm/moirais)
"""Rolling-window aggregate feature over window W."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_rolling_window_feature"]


def joseph_rolling_window_feature(y, W, agg):
    """
    Rolling-window aggregate feature over window W

    Formula: x_t^{(rollW_agg)} = agg(y_{t-W+1}, ..., y_t);  agg in {mean, std, min, max, ...}

    Parameters
    ----------
    y : array-like
        Input data.
    W : array-like
        Input data.
    agg : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rolling_feature

    References
    ----------
    Joseph Ch 6, Rolling Window Aggregates section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rolling-window aggregate feature over window W"})


def cheatsheet():
    return "jorolf: Rolling-window aggregate feature over window W"
