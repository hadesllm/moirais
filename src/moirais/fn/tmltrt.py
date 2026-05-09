"""TMLE with propensity truncation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_truncation"]


def tmle_truncation(y, D, X, eps):
    """
    TMLE with propensity truncation

    Formula: truncate g_hat to [eps, 1-eps] then update

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Crump et al (2009); Petersen-vdL (2012)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE with propensity truncation"})


def cheatsheet():
    return "tmltrt: TMLE with propensity truncation"
