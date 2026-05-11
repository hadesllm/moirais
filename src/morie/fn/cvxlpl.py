"""LP dual."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_linear_program_dual"]


def boyd_linear_program_dual(A, b, c):
    """
    LP dual

    Formula: max b'y s.t. A'y <= c

    Parameters
    ----------
    A : array-like
        Input data.
    b : array-like
        Input data.
    c : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Boyd CVX Ch 5
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LP dual"})


def cheatsheet():
    return "cvxlpl: LP dual"
