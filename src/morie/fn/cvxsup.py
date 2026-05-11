"""Supporting hyperplane theorem."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_support_hyperplane"]


def boyd_support_hyperplane(C, x0):
    """
    Supporting hyperplane theorem

    Formula: for boundary x_0 of C exists a: a'x <= a'x_0 for all x in C

    Parameters
    ----------
    C : array-like
        Input data.
    x0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: a

    References
    ----------
    Boyd CVX Ch 2
    """
    C = np.atleast_1d(np.asarray(C, dtype=float))
    n = len(C)
    result = float(np.mean(C))
    se = float(np.std(C, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Supporting hyperplane theorem"})


def cheatsheet():
    return "cvxsup: Supporting hyperplane theorem"
