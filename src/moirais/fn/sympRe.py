"""SymPy simplify expression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sympy_simplify"]


def sympy_simplify(expr):
    """
    SymPy simplify expression

    Formula: normalize via various heuristics

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
    SymPy team (2017)
    """
    expr = np.atleast_1d(np.asarray(expr, dtype=float))
    n = len(expr)
    result = float(np.mean(expr))
    se = float(np.std(expr, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SymPy simplify expression"})


def cheatsheet():
    return "sympRe: SymPy simplify expression"
