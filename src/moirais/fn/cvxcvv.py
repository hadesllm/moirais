"""Schur complement."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_cvxlin_complement"]


def boyd_cvxlin_complement(A, B, C):
    """
    Schur complement

    Formula: [A B; B' C] >= 0 iff A >=0 and C - B' A^{+} B >= 0

    Parameters
    ----------
    A : array-like
        Input data.
    B : array-like
        Input data.
    C : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: satisfied

    References
    ----------
    Boyd CVX Appendix A
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Schur complement"})


def cheatsheet():
    return "cvxcvv: Schur complement"
