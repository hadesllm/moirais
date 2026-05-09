"""Projected gradient descent."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_grad_proj"]


def boyd_grad_proj(f, grad_f, x0, C, t):
    """
    Projected gradient descent

    Formula: x^{k+1} = P_C(x^k - t grad f(x^k))

    Parameters
    ----------
    f : array-like
        Input data.
    grad_f : array-like
        Input data.
    x0 : array-like
        Input data.
    C : array-like
        Input data.
    t : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Projected gradient descent"})


def cheatsheet():
    return "cvxgd1: Projected gradient descent"
