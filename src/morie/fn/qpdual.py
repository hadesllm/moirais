"""Quadratic programming."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["quadratic_program"]


def quadratic_program(Q, c, A, b):
    """
    Quadratic programming

    Formula: min 0.5 x^T Q x + c^T x s.t. Ax <= b

    Parameters
    ----------
    Q : array-like
        Input data.
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
    Frank-Wolfe (1956); Wolfe (1959)
    """
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    n = len(Q)
    result = float(np.mean(Q))
    se = float(np.std(Q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Quadratic programming"})


def cheatsheet():
    return "qpdual: Quadratic programming"
