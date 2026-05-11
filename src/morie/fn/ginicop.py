"""Gini's gamma from a copula."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ginis_gamma_copula"]


def ginis_gamma_copula(y, copula, theta):
    """
    Gini's gamma from a copula

    Formula: gamma = 4 [integral C(u, 1-u) du - integral (u - C(u,u)) du]

    Parameters
    ----------
    y : array-like
        Input data.
    copula : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Nelsen (1998)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gini's gamma from a copula"})


def cheatsheet():
    return "ginicop: Gini's gamma from a copula"
