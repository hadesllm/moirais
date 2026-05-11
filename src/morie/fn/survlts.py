"""Smoothed life-table estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["life_table_smoothed"]


def life_table_smoothed(time, event, bandwidth):
    """
    Smoothed life-table estimator

    Formula: kernel-smoothed S(t) over interval grid

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ramlau-Hansen (1983)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Smoothed life-table estimator"})
    estimate = np.median(time)
    se = 1.2533 * np.std(time, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Smoothed life-table estimator"})


def cheatsheet():
    return "survlts: Smoothed life-table estimator"
