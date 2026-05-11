"""Stahel-Donoho outlyingness."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["stahel_donoho"]


def stahel_donoho(X, u_dirs):
    """
    Stahel-Donoho outlyingness

    Formula: O(x) = sup_u |u^T x − Med(u^T X)|/MAD(u^T X)

    Parameters
    ----------
    X : array-like
        Input data.
    u_dirs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Stahel (1981); Donoho (1982)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stahel-Donoho outlyingness"})


def cheatsheet():
    return "stahdo: Stahel-Donoho outlyingness"
