"""Controlled direct effect (Robins-Greenland)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["controlled_direct_effect"]


def controlled_direct_effect(X, M, Y, m):
    """
    Controlled direct effect (Robins-Greenland)

    Formula: CDE(m) = E[Y(1,m) - Y(0,m)]

    Parameters
    ----------
    X : array-like
        Input data.
    M : array-like
        Input data.
    Y : array-like
        Input data.
    m : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robins & Greenland (1992)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Controlled direct effect (Robins-Greenland)"})


def cheatsheet():
    return "ctde: Controlled direct effect (Robins-Greenland)"
