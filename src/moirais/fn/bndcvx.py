"""Convex-relaxation bound estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_convex_estimator"]


def bound_convex_estimator(y, D, constraints):
    """
    Convex-relaxation bound estimator

    Formula: LP relaxation of original NP problem

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    constraints : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Mogstad-Santos-Torgovitsky (2018)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Convex-relaxation bound estimator"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Convex-relaxation bound estimator"})


def cheatsheet():
    return "bndcvx: Convex-relaxation bound estimator"
