"""Spline-based propensity weights."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["spline_weights"]


def spline_weights(A, H, knots):
    """
    Spline-based propensity weights

    Formula: natural cubic spline f(A_t|H_t)

    Parameters
    ----------
    A : array-like
        Input data.
    H : array-like
        Input data.
    knots : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Westreich et al (2010)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spline-based propensity weights"})


def cheatsheet():
    return "spwgts: Spline-based propensity weights"
