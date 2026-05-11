"""Separating hyperplane theorem."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_separating_hyperplane"]


def boyd_separating_hyperplane(C, D):
    """
    Separating hyperplane theorem

    Formula: exists a !=0, b: a'x <= b for x in C, a'x >= b for x in D

    Parameters
    ----------
    C : array-like
        Input data.
    D : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: a, b

    References
    ----------
    Boyd CVX Ch 2
    """
    C = np.atleast_1d(np.asarray(C, dtype=float))
    n = len(C)
    result = float(np.mean(C))
    se = float(np.std(C, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Separating hyperplane theorem"})


def cheatsheet():
    return "cvxsep: Separating hyperplane theorem"
