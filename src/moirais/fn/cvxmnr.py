"""Minimax problem."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_minimax"]


def boyd_minimax(f):
    """
    Minimax problem

    Formula: min max_i f_i(x)

    Parameters
    ----------
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x

    References
    ----------
    Boyd CVX Ch 4
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Minimax problem"})


def cheatsheet():
    return "cvxmnr: Minimax problem"
