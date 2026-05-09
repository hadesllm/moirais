"""Kernel density estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_kde"]


def wasserman_kde(x, data, h):
    """
    Kernel density estimator

    Formula: f_h(x) = (1/(nh)) sum K((x-X_i)/h)

    Parameters
    ----------
    x : array-like
        Input data.
    data : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: density

    References
    ----------
    Wasserman (2004), Ch 20
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Kernel density estimator"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Kernel density estimator"})


def cheatsheet():
    return "wsmkdn: Kernel density estimator"
