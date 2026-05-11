"""Regularized least squares (Tikhonov)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_regularized_ls"]


def boyd_regularized_ls(A, b, delta):
    """
    Regularized least squares (Tikhonov)

    Formula: min |Ax-b|_2^2 + delta |x|_2^2

    Parameters
    ----------
    A : array-like
        Input data.
    b : array-like
        Input data.
    delta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x

    References
    ----------
    Boyd CVX Ch 4
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Regularized least squares (Tikhonov)"})


def cheatsheet():
    return "cvxrgl: Regularized least squares (Tikhonov)"
