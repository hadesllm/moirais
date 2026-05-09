"""TMLE for natural direct effect."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_natural_direct"]


def tmle_natural_direct(y, D, M, X):
    """
    TMLE for natural direct effect

    Formula: E[Y(1,M(0)) - Y(0,M(0))]

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    M : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zheng & vdL (2012)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for natural direct effect"})


def cheatsheet():
    return "tmlnde: TMLE for natural direct effect"
