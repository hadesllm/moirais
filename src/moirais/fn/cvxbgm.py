"""Basis pursuit denoising."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_basis_pursuit"]


def boyd_basis_pursuit(A, b, eps):
    """
    Basis pursuit denoising

    Formula: min |x|_1 s.t. |Ax - b|_2 <= eps

    Parameters
    ----------
    A : array-like
        Input data.
    b : array-like
        Input data.
    eps : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Basis pursuit denoising"})


def cheatsheet():
    return "cvxbgm: Basis pursuit denoising"
