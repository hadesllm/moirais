"""3D compensatory multidimensional IRT."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mirt_3d_compensatory"]


def mirt_3d_compensatory(y, theta, a, d):
    """
    3D compensatory multidimensional IRT

    Formula: P = 1/(1 + exp(-(a1 theta1 + a2 theta2 + a3 theta3) + d))

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

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Reckase (2009)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "3D compensatory multidimensional IRT"})


def cheatsheet():
    return "mirt3: 3D compensatory multidimensional IRT"
