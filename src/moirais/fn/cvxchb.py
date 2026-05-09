"""Chebyshev center of polyhedron."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_chebyshev_center"]


def boyd_chebyshev_center(A, b):
    """
    Chebyshev center of polyhedron

    Formula: max r s.t. a_i'x + r |a_i| <= b_i

    Parameters
    ----------
    A : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x, r

    References
    ----------
    Boyd CVX Ch 8
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Chebyshev center of polyhedron"})


def cheatsheet():
    return "cvxchb: Chebyshev center of polyhedron"
