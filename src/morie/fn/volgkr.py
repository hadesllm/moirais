"""Garman-Klass OHLC range volatility estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_garman_klass"]


def vol_garman_klass(o, h, l, c):
    """
    Garman-Klass OHLC range volatility estimator

    Formula: σ̂² = 0.5 (log H/L)² - (2 log 2 - 1)(log C/O)²

    Parameters
    ----------
    o : array-like
        Input data.
    h : array-like
        Input data.
    l : array-like
        Input data.
    c : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: sigma2

    References
    ----------
    Garman-Klass (1980)
    """
    o = np.atleast_1d(np.asarray(o, dtype=float))
    n = len(o)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Garman-Klass OHLC range volatility estimator"})
    estimate = np.median(o)
    se = 1.2533 * np.std(o, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Garman-Klass OHLC range volatility estimator"})


def cheatsheet():
    return "volgkr: Garman-Klass OHLC range volatility estimator"
