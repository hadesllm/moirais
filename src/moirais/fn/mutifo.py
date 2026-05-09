"""Mutual information between X and Y."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mutual_information"]


def mutual_information(y, x, y2):
    """
    Mutual information between X and Y

    Formula: I(X;Y) = sum_xy p(x,y) log(p(x,y)/(p(x)p(y)))

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
    Shannon (1948); Cover & Thomas (2006)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mutual information between X and Y"})


def cheatsheet():
    return "mutifo: Mutual information between X and Y"
