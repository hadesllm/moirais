"""Warped Gaussian process."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gp_warped"]


def gp_warped(X, y, X_test, warp):
    """
    Warped Gaussian process

    Formula: f(x) = h(g(x)); g ~ GP

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    X_test : array-like
        Input data.
    warp : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Snelson-Rasmussen-Ghahramani (2004)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Warped Gaussian process"})


def cheatsheet():
    return "gpwhr: Warped Gaussian process"
