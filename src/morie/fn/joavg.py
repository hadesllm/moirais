# morie.fn -- function file (hadesllm/morie)
"""Average forecast: mean of the historical series."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_average_forecast"]


def joseph_average_forecast(y, horizon):
    """
    Average forecast: mean of the historical series

    Formula: y_hat_{T+h} = mean(y_1, ..., y_T)

    Parameters
    ----------
    y : array-like
        Input data.
    horizon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_hat

    References
    ----------
    Joseph Ch 4, Average Forecast section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Average forecast: mean of the historical series"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Average forecast: mean of the historical series"})


def cheatsheet():
    return "joavg: Average forecast: mean of the historical series"
