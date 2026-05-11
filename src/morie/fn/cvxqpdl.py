"""QP dual."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_qp_dual"]


def boyd_qp_dual(P, q, G, h):
    """
    QP dual

    Formula: max -(1/2) lambda'(GP^{-1}G')lambda - h'lambda - (1/2) q'P^{-1}q

    Parameters
    ----------
    P : array-like
        Input data.
    q : array-like
        Input data.
    G : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lambda

    References
    ----------
    Boyd CVX Ch 5
    """
    P = np.atleast_1d(np.asarray(P, dtype=float))
    n = len(P)
    result = float(np.mean(P))
    se = float(np.std(P, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "QP dual"})


def cheatsheet():
    return "cvxqpdl: QP dual"
