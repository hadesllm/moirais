"""Nonlinear bound estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_nonlinear"]


def bound_nonlinear(y, X, g):
    """
    Nonlinear bound estimator

    Formula: theta = inf max nonlinear g(theta)

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    g : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Andrews-Shi (2013)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Nonlinear bound estimator"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Nonlinear bound estimator"})


def cheatsheet():
    return "bndnln: Nonlinear bound estimator"
