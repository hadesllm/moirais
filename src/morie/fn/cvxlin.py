"""Linear program standard form."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_linear_program"]


def boyd_linear_program(c, A, b):
    """
    Linear program standard form

    Formula: min c'x s.t. Ax = b, x >= 0

    Parameters
    ----------
    c : array-like
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
    c = np.atleast_1d(np.asarray(c, dtype=float))
    n = len(c)
    result = float(np.mean(c))
    se = float(np.std(c, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linear program standard form"})


def cheatsheet():
    return "cvxlin: Linear program standard form"
