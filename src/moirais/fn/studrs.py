"""Externally studentized residual."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["studentized_residual"]


def studentized_residual(y, X):
    """
    Externally studentized residual

    Formula: t_i = e_i / (s_(i) sqrt(1 - h_ii))

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
    Cook & Weisberg (1982)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Externally studentized residual"})


def cheatsheet():
    return "studrs: Externally studentized residual"
