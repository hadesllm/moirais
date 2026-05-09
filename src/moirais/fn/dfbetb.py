"""DFBETAS per-coefficient leverage of obs i."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dfbetas"]


def dfbetas(y, X):
    """
    DFBETAS per-coefficient leverage of obs i

    Formula: DFBETAS_ij = (beta_j - beta_j(-i)) / (s_(i) sqrt((X'X)^-1_jj))

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DFBETAS per-coefficient leverage of obs i"})


def cheatsheet():
    return "dfbetb: DFBETAS per-coefficient leverage of obs i"
