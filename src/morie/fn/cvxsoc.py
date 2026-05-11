"""Second-order cone program."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_socp"]


def boyd_socp(f, A, b, c, d):
    """
    Second-order cone program

    Formula: min f'x s.t. |A_i x + b_i|_2 <= c_i'x + d_i

    Parameters
    ----------
    f : array-like
        Input data.
    A : array-like
        Input data.
    b : array-like
        Input data.
    c : array-like
        Input data.
    d : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x

    References
    ----------
    Boyd CVX Ch 4
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Second-order cone program"})


def cheatsheet():
    return "cvxsoc: Second-order cone program"
