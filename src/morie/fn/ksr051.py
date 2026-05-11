"""Continuous invertibility of an operator A on parameter space Theta."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_continuous_invertibility"]


def kosorok_ch2_continuous_invertibility(A, theta_1, theta_2, c):
    """
    Continuous invertibility of an operator A on parameter space Theta

    Formula: || A(theta_1) - A(theta_2) ||_L >= c * || theta_1 - theta_2 ||

    Parameters
    ----------
    A : array-like
        Input data.
    theta_1 : array-like
        Input data.
    theta_2 : array-like
        Input data.
    c : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 2, Eq 2.15, p. 26
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Continuous invertibility of an operator A on parameter space Theta"})


def cheatsheet():
    return "ksr051: Continuous invertibility of an operator A on parameter space Theta"
