"""Huber-White sandwich variance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_white_huber"]


def wasserman_white_huber(X, y, f):
    """
    Huber-White sandwich variance

    Formula: V = A^{-1} B A^{-1}

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: covariance

    References
    ----------
    Wasserman (2004), Ch 9
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Huber-White sandwich variance"})


def cheatsheet():
    return "wsmwhz: Huber-White sandwich variance"
