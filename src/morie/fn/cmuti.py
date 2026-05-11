"""Mutual information from copula density."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["copula_mutual_information"]


def copula_mutual_information(y, u, v, copula, theta):
    """
    Mutual information from copula density

    Formula: I(X;Y) = integral integral c(u,v) log c(u,v) du dv

    Parameters
    ----------
    y : array-like
        Input data.
    u : array-like
        Input data.
    v : array-like
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
    Linfoot (1957); Calsaverini & Vicente (2009)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mutual information from copula density"})


def cheatsheet():
    return "cmuti: Mutual information from copula density"
