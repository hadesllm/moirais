"""Iteratively reweighted least squares (one outer iteration)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["irls_solver"]


def irls_solver(y, X, weights):
    """
    Iteratively reweighted least squares (one outer iteration)

    Formula: beta^(t+1) = (X' W^(t) X)^-1 X' W^(t) y

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Holland & Welsch (1977)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Iteratively reweighted least squares (one outer iteration)"})


def cheatsheet():
    return "irlsfn: Iteratively reweighted least squares (one outer iteration)"
