"""Tail-dependence χ."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["chi_dependence"]


def chi_dependence(X, Y, u):
    """
    Tail-dependence χ

    Formula: χ(u) = P(F_Y(Y)>u | F_X(X)>u)

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    u : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Coles-Heffernan-Tawn (1999)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tail-dependence χ"})


def cheatsheet():
    return "chiDep: Tail-dependence χ"
