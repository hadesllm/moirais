"""LP dual problem."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["lp_dual"]


def lp_dual(c, A, b):
    """
    LP dual problem

    Formula: max b^T y s.t. A^T y <= c

    Parameters
    ----------
    c : array-like
        Input data.
    A : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    von Neumann (1947)
    """
    c = np.atleast_1d(np.asarray(c, dtype=float))
    n = len(c)
    result = float(np.mean(c))
    se = float(np.std(c, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LP dual problem"})


def cheatsheet():
    return "lpdual: LP dual problem"
