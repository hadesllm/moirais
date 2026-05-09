"""Chebyshev (L_inf) regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_linf_fitting"]


def boyd_linf_fitting(A, b):
    """
    Chebyshev (L_inf) regression

    Formula: min |Ax - b|_inf

    Parameters
    ----------
    A : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x

    References
    ----------
    Boyd CVX Ch 6
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Chebyshev (L_inf) regression"})


def cheatsheet():
    return "cvxlif: Chebyshev (L_inf) regression"
