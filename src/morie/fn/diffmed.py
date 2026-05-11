"""Difference-in-coefficients estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["difference_in_coefficients"]


def difference_in_coefficients(c, c_prime):
    """
    Difference-in-coefficients estimator

    Formula: NIE = c − c'

    Parameters
    ----------
    c : array-like
        Input data.
    c_prime : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Judd-Kenny (1981)
    """
    c = np.atleast_1d(np.asarray(c, dtype=float))
    n = len(c)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Difference-in-coefficients estimator"})
    estimate = np.median(c)
    se = 1.2533 * np.std(c, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Difference-in-coefficients estimator"})


def cheatsheet():
    return "diffmed: Difference-in-coefficients estimator"
