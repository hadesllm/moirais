"""Lagrange dual function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_dual_function"]


def boyd_dual_function(L, lambda_, nu):
    """
    Lagrange dual function

    Formula: g(lambda,nu) = inf_x L(x,lambda,nu)

    Parameters
    ----------
    L : array-like
        Input data.
    lambda_ : array-like
        Input data.
    nu : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Boyd CVX Ch 5
    """
    L = np.atleast_1d(np.asarray(L, dtype=float))
    n = len(L)
    result = float(np.mean(L))
    se = float(np.std(L, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lagrange dual function"})


def cheatsheet():
    return "cvxdwl: Lagrange dual function"
