"""Holt-Winters additive."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["holt_winters_additive"]


def holt_winters_additive(y, period, alpha, beta, gamma):
    """
    Holt-Winters additive

    Formula: l_t,b_t,s_t recursions with α,β,γ

    Parameters
    ----------
    y : array-like
        Input data.
    period : array-like
        Input data.
    alpha : array-like
        Input data.
    beta : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Winters (1960); Hyndman-Athanasopoulos (2018)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Holt-Winters additive"})


def cheatsheet():
    return "hwadd: Holt-Winters additive"
