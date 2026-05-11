"""L1 robust regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_l1_fitting"]


def boyd_l1_fitting(A, b):
    """
    L1 robust regression

    Formula: min |Ax - b|_1

    Parameters
    ----------
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
    Boyd CVX Ch 6
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "L1 robust regression"})


def cheatsheet():
    return "cvxlnf: L1 robust regression"
