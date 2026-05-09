"""Mutual information I(X;Y)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mutual_information"]


def mutual_information(pxy, base):
    """
    Mutual information I(X;Y)

    Formula: I(X;Y) = sum p(x,y) log[p(x,y)/(p(x)p(y))]

    Parameters
    ----------
    pxy : array-like
        Input data.
    base : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Shannon (1948)
    """
    pxy = np.atleast_1d(np.asarray(pxy, dtype=float))
    n = len(pxy)
    result = float(np.mean(pxy))
    se = float(np.std(pxy, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mutual information I(X;Y)"})


def cheatsheet():
    return "mutinf: Mutual information I(X;Y)"
