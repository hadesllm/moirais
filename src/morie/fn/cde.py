"""Controlled direct effect."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["controlled_direct_effect"]


def controlled_direct_effect(Y, X, M, m):
    """
    Controlled direct effect

    Formula: E[Y(1, m) − Y(0, m)] for fixed m

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.
    M : array-like
        Input data.
    m : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robins-Greenland (1992)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Controlled direct effect"})


def cheatsheet():
    return "cde: Controlled direct effect"
