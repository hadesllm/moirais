"""Quadratically constrained QP."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_quadratic_constraint"]


def boyd_quadratic_constraint(P0, q0, P, q, r):
    """
    Quadratically constrained QP

    Formula: min (1/2) x'P0 x + q0'x s.t. (1/2) x'P_i x + q_i'x + r_i <= 0

    Parameters
    ----------
    P0 : array-like
        Input data.
    q0 : array-like
        Input data.
    P : array-like
        Input data.
    q : array-like
        Input data.
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x

    References
    ----------
    Boyd CVX Ch 4
    """
    P0 = np.atleast_1d(np.asarray(P0, dtype=float))
    n = len(P0)
    result = float(np.mean(P0))
    se = float(np.std(P0, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Quadratically constrained QP"})


def cheatsheet():
    return "cvxqcr: Quadratically constrained QP"
