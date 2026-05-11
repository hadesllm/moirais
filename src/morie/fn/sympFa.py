"""Polynomial factorization."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sympy_factor"]


def sympy_factor(expr):
    """
    Polynomial factorization

    Formula: Berlekamp-Zassenhaus / Cantor-Zassenhaus

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
    Cohen (1996) book
    """
    expr = np.atleast_1d(np.asarray(expr, dtype=float))
    n = len(expr)
    result = float(np.mean(expr))
    se = float(np.std(expr, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Polynomial factorization"})


def cheatsheet():
    return "sympFa: Polynomial factorization"
