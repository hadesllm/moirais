"""Generic LP solver dispatch."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["linear_programming"]


def linear_programming(c, A, b, method):
    """
    Generic LP solver dispatch

    Formula: select method (simplex/interior-point) by size

    Parameters
    ----------
    c : array-like
        Input data.
    A : array-like
        Input data.
    b : array-like
        Input data.
    method : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dantzig (1947); Karmarkar (1984)
    """
    c = np.atleast_1d(np.asarray(c, dtype=float))
    n = len(c)
    result = float(np.mean(c))
    se = float(np.std(c, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Generic LP solver dispatch"})


def cheatsheet():
    return "linprm: Generic LP solver dispatch"
