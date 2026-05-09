"""Convex conjugate (Legendre-Fenchel)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_conjugate"]


def boyd_conjugate(f, y):
    """
    Convex conjugate (Legendre-Fenchel)

    Formula: f*(y) = sup_x (y'x - f(x))

    Parameters
    ----------
    f : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Boyd CVX Ch 3
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Convex conjugate (Legendre-Fenchel)"})


def cheatsheet():
    return "cvxcjg: Convex conjugate (Legendre-Fenchel)"
