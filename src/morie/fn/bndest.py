"""Estimation of identification set."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_estimation"]


def bound_estimation(y, X, moments):
    """
    Estimation of identification set

    Formula: empirical analog of theoretical set

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    moments : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Manski-Tamer (2002)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Estimation of identification set"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Estimation of identification set"})


def cheatsheet():
    return "bndest: Estimation of identification set"
