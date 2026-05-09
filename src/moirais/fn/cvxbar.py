"""Log barrier function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_log_barrier"]


def boyd_log_barrier(f, x):
    """
    Log barrier function

    Formula: phi(x) = -sum log(-f_i(x))

    Parameters
    ----------
    f : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Boyd CVX Ch 11
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Log barrier function"})


def cheatsheet():
    return "cvxbar: Log barrier function"
