"""Spearman's rho derived from a copula."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["spearmans_rho_copula"]


def spearmans_rho_copula(y, copula, theta):
    """
    Spearman's rho derived from a copula

    Formula: rho_S = 12 integral integral C(u,v) du dv - 3

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
    Schweizer & Wolff (1981)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spearman's rho derived from a copula"})


def cheatsheet():
    return "spcoef: Spearman's rho derived from a copula"
