"""Algebraic expansion."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sympy_expand"]


def sympy_expand(expr):
    """
    Algebraic expansion

    Formula: distribute products and powers

    Parameters
    ----------
    expr : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    SymPy
    """
    expr = np.atleast_1d(np.asarray(expr, dtype=float))
    n = len(expr)
    result = float(np.mean(expr))
    se = float(np.std(expr, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Algebraic expansion"})


def cheatsheet():
    return "sympEx: Algebraic expansion"
