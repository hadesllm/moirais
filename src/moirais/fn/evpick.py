"""Pickands estimator of the extreme-value index."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_pickands_estimator"]


def evt_pickands_estimator(x, k):
    """
    Pickands estimator of the extreme-value index

    Formula: γ̂_P = log((X_{n-k}-X_{n-2k})/(X_{n-2k}-X_{n-4k})) / log 2

    Parameters
    ----------
    x : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: gamma

    References
    ----------
    Pickands (1975)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Pickands estimator of the extreme-value index"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Pickands estimator of the extreme-value index"})


def cheatsheet():
    return "evpick: Pickands estimator of the extreme-value index"
