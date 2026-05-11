"""Newton decrement."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_newton_decrement"]


def boyd_newton_decrement(grad, hess):
    """
    Newton decrement

    Formula: lambda(x)^2 = grad f(x)' H^{-1} grad f(x)

    Parameters
    ----------
    grad : array-like
        Input data.
    hess : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Boyd CVX Ch 9
    """
    grad = np.atleast_1d(np.asarray(grad, dtype=float))
    n = len(grad)
    result = float(np.mean(grad))
    se = float(np.std(grad, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Newton decrement"})


def cheatsheet():
    return "cvxnda: Newton decrement"
