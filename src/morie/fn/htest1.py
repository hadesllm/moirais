"""Horvitz-Thompson estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["horvitz_thompson"]


def horvitz_thompson(y, pi):
    """
    Horvitz-Thompson estimator

    Formula: sum y_i / pi_i

    Parameters
    ----------
    y : array-like
        Input data.
    pi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Horvitz-Thompson (1952)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Horvitz-Thompson estimator"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Horvitz-Thompson estimator"})


def cheatsheet():
    return "htest1: Horvitz-Thompson estimator"
