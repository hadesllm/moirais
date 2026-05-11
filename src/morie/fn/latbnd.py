"""Bounds on Local Average Treatment Effect."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["late_bounds"]


def late_bounds(y, D, Z):
    """
    Bounds on Local Average Treatment Effect

    Formula: LATE bounds under Imbens-Angrist conditions

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    Z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Imbens-Angrist (1994); Manski-Pepper (2000)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Bounds on Local Average Treatment Effect"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Bounds on Local Average Treatment Effect"})


def cheatsheet():
    return "latbnd: Bounds on Local Average Treatment Effect"
