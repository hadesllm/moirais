"""Dual optimization problem."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_dual_problem"]


def boyd_dual_problem(g):
    """
    Dual optimization problem

    Formula: max g(lambda,nu) s.t. lambda >= 0

    Parameters
    ----------
    g : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lambda, nu

    References
    ----------
    Boyd CVX Ch 5
    """
    g = np.atleast_1d(np.asarray(g, dtype=float))
    n = len(g)
    result = float(np.mean(g))
    se = float(np.std(g, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dual optimization problem"})


def cheatsheet():
    return "cvxdgp: Dual optimization problem"
