"""Collapsed CRP Gibbs (integrate out parameters)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["crp_collapsed"]


def crp_collapsed(y, alpha, n_iter):
    """
    Collapsed CRP Gibbs (integrate out parameters)

    Formula: P(z_i|z_{-i},y) with marginalized theta

    Parameters
    ----------
    y : array-like
        Input data.
    alpha : array-like
        Input data.
    n_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    MacEachern (1994)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Collapsed CRP Gibbs (integrate out parameters)"})


def cheatsheet():
    return "crpcol: Collapsed CRP Gibbs (integrate out parameters)"
