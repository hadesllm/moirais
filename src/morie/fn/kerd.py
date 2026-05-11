"""Kernel density estimate."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kernel_density_fda"]


def kernel_density_fda(x, h, kernel):
    """
    Kernel density estimate

    Formula: f̂(x) = (1/nh) sum K((x-x_i)/h)

    Parameters
    ----------
    x : array-like
        Input data.
    h : array-like
        Input data.
    kernel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Silverman (1986)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Kernel density estimate"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Kernel density estimate"})


def cheatsheet():
    return "kerd: Kernel density estimate"
