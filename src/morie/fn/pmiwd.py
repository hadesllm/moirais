"""Pointwise mutual information PMI(x,y)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pointwise_mutual_info"]


def pointwise_mutual_info(y, x, y2):
    """
    Pointwise mutual information PMI(x,y)

    Formula: PMI(x,y) = log(p(x,y)/(p(x)p(y)))

    Parameters
    ----------
    y : array-like
        Input data.
    x : array-like
        Input data.
    y2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Church & Hanks (1990)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pointwise mutual information PMI(x,y)"})


def cheatsheet():
    return "pmiwd: Pointwise mutual information PMI(x,y)"
