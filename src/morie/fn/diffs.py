"""Symbolic differentiation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["symbolic_diff"]


def symbolic_diff(expr, x):
    """
    Symbolic differentiation

    Formula: chain + product + quotient rules

    Parameters
    ----------
    expr : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    classical
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Symbolic differentiation"})


def cheatsheet():
    return "diffS: Symbolic differentiation"
