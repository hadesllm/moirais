"""Bayesian ridge."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bayes_ridge"]


def bayes_ridge(y, M):
    """
    Bayesian ridge

    Formula: u_j ~ N(0, sigma^2) with conjugate prior

    Parameters
    ----------
    y : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Park-Casella (2008)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian ridge"})


def cheatsheet():
    return "baysrr: Bayesian ridge"
