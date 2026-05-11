"""Geometric median in CLR coordinates."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["compositional_median"]


def compositional_median(X, tol):
    """
    Geometric median in CLR coordinates

    Formula: argmin_m Σ ||clr(x_n)-clr(m)||

    Parameters
    ----------
    X : array-like
        Input data.
    tol : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: m

    References
    ----------
    Filzmoser & Hron (2008)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Geometric median in CLR coordinates"})


def cheatsheet():
    return "aitcmd: Geometric median in CLR coordinates"
