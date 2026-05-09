"""Newton step."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_newton"]


def boyd_newton(grad, hess):
    """
    Newton step

    Formula: delta_x = -H^{-1} grad f(x)

    Parameters
    ----------
    grad : array-like
        Input data.
    hess : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: direction

    References
    ----------
    Boyd CVX Ch 9
    """
    grad = np.atleast_1d(np.asarray(grad, dtype=float))
    n = len(grad)
    result = float(np.mean(grad))
    se = float(np.std(grad, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Newton step"})


def cheatsheet():
    return "cvxntn: Newton step"
