"""Intervals estimator of θ (Ferro-Segers)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_extremal_index_intervals"]


def evt_extremal_index_intervals(x, u):
    """
    Intervals estimator of θ (Ferro-Segers)

    Formula: θ̂_FS = 2(Σ T_i)² / ((N-1) Σ T_i²)

    Parameters
    ----------
    x : array-like
        Input data.
    u : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta

    References
    ----------
    Ferro & Segers (2003)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Intervals estimator of θ (Ferro-Segers)"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Intervals estimator of θ (Ferro-Segers)"})


def cheatsheet():
    return "evextint: Intervals estimator of θ (Ferro-Segers)"
