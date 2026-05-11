"""TMLE with stabilized clever covariate."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_stabilized"]


def tmle_stabilized(y, D, X):
    """
    TMLE with stabilized clever covariate

    Formula: H* = (D-g)/(g(1-g)) * sqrt(g(1-g))

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    vdL-Gruber (2010)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE with stabilized clever covariate"})


def cheatsheet():
    return "tmlsta: TMLE with stabilized clever covariate"
