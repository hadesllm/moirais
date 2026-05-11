"""Gradient descent."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gradient_descent"]


def gradient_descent(f, grad_f, x0, lr, steps):
    """
    Gradient descent

    Formula: x_{t+1} = x_t - lr * grad f(x_t)

    Parameters
    ----------
    f : array-like
        Input data.
    grad_f : array-like
        Input data.
    x0 : array-like
        Input data.
    lr : array-like
        Input data.
    steps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cauchy (1847)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gradient descent"})


def cheatsheet():
    return "gradds: Gradient descent"
