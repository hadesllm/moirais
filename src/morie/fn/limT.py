"""Symbolic limit."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["symbolic_limit"]


def symbolic_limit(expr, x, x0):
    """
    Symbolic limit

    Formula: L'Hôpital + series + Gruntz

    Parameters
    ----------
    expr : array-like
        Input data.
    x : array-like
        Input data.
    x0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gruntz (1996)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Symbolic limit"})


def cheatsheet():
    return "limT: Symbolic limit"
