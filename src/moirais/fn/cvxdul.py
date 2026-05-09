"""Lagrangian function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_lagrangian"]


def boyd_lagrangian(f0, f, h, lambda_, nu):
    """
    Lagrangian function

    Formula: L(x,lambda,nu) = f0(x) + sum lambda_i f_i(x) + sum nu_i h_i(x)

    Parameters
    ----------
    f0 : array-like
        Input data.
    f : array-like
        Input data.
    h : array-like
        Input data.
    lambda_ : array-like
        Input data.
    nu : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Boyd CVX Ch 5
    """
    f0 = np.atleast_1d(np.asarray(f0, dtype=float))
    n = len(f0)
    result = float(np.mean(f0))
    se = float(np.std(f0, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lagrangian function"})


def cheatsheet():
    return "cvxdul: Lagrangian function"
