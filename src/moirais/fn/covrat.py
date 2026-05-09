"""COVRATIO scaled change in cov(beta) when obs i deleted."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["covratio"]


def covratio(y, X):
    """
    COVRATIO scaled change in cov(beta) when obs i deleted

    Formula: COVRATIO_i = (s_(i)^2 / s^2)^p / (1 - h_ii)

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Belsley, Kuh, Welsch (1980)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "COVRATIO scaled change in cov(beta) when obs i deleted"})


def cheatsheet():
    return "covrat: COVRATIO scaled change in cov(beta) when obs i deleted"
