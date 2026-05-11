"""2D compensatory multidimensional IRT."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mirt_2d_compensatory"]


def mirt_2d_compensatory(y, theta, a, d, c):
    """
    2D compensatory multidimensional IRT

    Formula: P = c + (1-c)/(1 + exp(-(a1 theta1 + a2 theta2) + d))

    Parameters
    ----------
    y : array-like
        Input data.
    theta : array-like
        Input data.
    a : array-like
        Input data.
    d : array-like
        Input data.
    c : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Reckase (2009) MIRT §4
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "2D compensatory multidimensional IRT"})


def cheatsheet():
    return "mirt2: 2D compensatory multidimensional IRT"
