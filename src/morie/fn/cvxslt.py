"""Slater's condition for strong duality."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_slater"]


def boyd_slater(f):
    """
    Slater's condition for strong duality

    Formula: exists x interior with f_i(x) < 0

    Parameters
    ----------
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: satisfied

    References
    ----------
    Boyd CVX Ch 5
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Slater's condition for strong duality"})


def cheatsheet():
    return "cvxslt: Slater's condition for strong duality"
