"""Pointwise almost sure convergence of empirical to true distribution function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_law_large_numbers_pointwise"]


def kosorok_ch2_law_large_numbers_pointwise(F_n, F, t):
    """
    Pointwise almost sure convergence of empirical to true distribution function

    Formula: F_n(t) -> F(t) almost surely for each t in R

    Parameters
    ----------
    F_n : array-like
        Input data.
    F : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 2, Eq 2.2, p. 10
    """
    F_n = np.atleast_1d(np.asarray(F_n, dtype=float))
    n = len(F_n)
    result = float(np.mean(F_n))
    se = float(np.std(F_n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pointwise almost sure convergence of empirical to true distribution function"})


def cheatsheet():
    return "ksr027: Pointwise almost sure convergence of empirical to true distribution function"
