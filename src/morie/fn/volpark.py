"""Parkinson high-low range volatility estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_parkinson_range"]


def vol_parkinson_range(high, low):
    """
    Parkinson high-low range volatility estimator

    Formula: σ̂² = (1/(4 log 2 n)) Σ (log(H_t/L_t))²

    Parameters
    ----------
    high : array-like
        Input data.
    low : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: sigma2

    References
    ----------
    Parkinson (1980)
    """
    high = np.atleast_1d(np.asarray(high, dtype=float))
    n = len(high)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Parkinson high-low range volatility estimator"})
    estimate = np.median(high)
    se = 1.2533 * np.std(high, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Parkinson high-low range volatility estimator"})


def cheatsheet():
    return "volpark: Parkinson high-low range volatility estimator"
