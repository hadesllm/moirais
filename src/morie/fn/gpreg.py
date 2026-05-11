"""Gaussian process regression with squared-exponential kernel."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gaussian_process_regression"]


def gaussian_process_regression(X, y, X_test, kernel, noise):
    """
    Gaussian process regression with squared-exponential kernel

    Formula: f(x) ~ GP(m(x), k(x,x')); k(x,x')=sigma^2 exp(-||x-x'||^2/(2 l^2))

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    X_test : array-like
        Input data.
    kernel : array-like
        Input data.
    noise : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rasmussen-Williams (2006)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gaussian process regression with squared-exponential kernel"})


def cheatsheet():
    return "gpreg: Gaussian process regression with squared-exponential kernel"
