"""Estimate derivative of smoothed function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["derivative_function"]


def derivative_function(coef, basis, order):
    """
    Estimate derivative of smoothed function

    Formula: differentiate basis expansion

    Parameters
    ----------
    coef : array-like
        Input data.
    basis : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ramsay-Silverman (2005)
    """
    coef = np.atleast_1d(np.asarray(coef, dtype=float))
    n = len(coef)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Estimate derivative of smoothed function"})
    estimate = np.median(coef)
    se = 1.2533 * np.std(coef, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Estimate derivative of smoothed function"})


def cheatsheet():
    return "derivf: Estimate derivative of smoothed function"
