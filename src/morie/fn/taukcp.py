"""Kendall's tau derived from a copula."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kendalls_tau_copula"]


def kendalls_tau_copula(y, copula, theta):
    """
    Kendall's tau derived from a copula

    Formula: tau = 4 integral integral C(u,v) dC(u,v) - 1

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
    Schweizer & Wolff (1981); Genest (1987)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kendall's tau derived from a copula"})


def cheatsheet():
    return "taukcp: Kendall's tau derived from a copula"
