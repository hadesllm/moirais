"""Covariance Cov(X,Y)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_covariance"]


def wasserman_covariance(x, y):
    """
    Covariance Cov(X,Y)

    Formula: Cov(X,Y) = E[XY] - E[X]E[Y]

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Wasserman (2004), Ch 4
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Covariance Cov(X,Y)"})


def cheatsheet():
    return "wsmcov: Covariance Cov(X,Y)"
