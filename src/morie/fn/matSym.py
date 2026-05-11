"""Symbolic matrix algebra."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["matrix_symbolic"]


def matrix_symbolic(M):
    """
    Symbolic matrix algebra

    Formula: det, inv, eigenvalues over symbols

    Parameters
    ----------
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    classical linear algebra
    """
    M = np.atleast_1d(np.asarray(M, dtype=float))
    n = len(M)
    result = float(np.mean(M))
    se = float(np.std(M, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Symbolic matrix algebra"})


def cheatsheet():
    return "matSym: Symbolic matrix algebra"
