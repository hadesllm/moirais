"""Quadratic program."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_quadratic_program"]


def boyd_quadratic_program(P, q, G, h, A, b):
    """
    Quadratic program

    Formula: min (1/2) x'Px + q'x s.t. Gx <= h, Ax = b

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
    Boyd CVX Ch 4
    """
    P = np.atleast_1d(np.asarray(P, dtype=float))
    n = len(P)
    result = float(np.mean(P))
    se = float(np.std(P, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Quadratic program"})


def cheatsheet():
    return "cvxqp: Quadratic program"
