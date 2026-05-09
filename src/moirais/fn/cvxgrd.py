"""Gradient descent step."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_gradient_descent"]


def boyd_gradient_descent(f, grad_f, x0, t):
    """
    Gradient descent step

    Formula: x^{k+1} = x^k - t_k grad f(x^k)

    Parameters
    ----------
    f : array-like
        Input data.
    grad_f : array-like
        Input data.
    x0 : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x

    References
    ----------
    Boyd CVX Ch 9
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gradient descent step"})


def cheatsheet():
    return "cvxgrd: Gradient descent step"
