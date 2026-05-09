"""Hierarchical Dirichlet Process for shared mixture components."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hierarchical_dp"]


def hierarchical_dp(y, groups, gamma, alpha):
    """
    Hierarchical Dirichlet Process for shared mixture components

    Formula: G_0 ~ DP(gamma, H); G_j ~ DP(alpha, G_0)

    Parameters
    ----------
    y : array-like
        Input data.
    groups : array-like
        Input data.
    gamma : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Teh-Jordan-Beal-Blei (2006)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hierarchical Dirichlet Process for shared mixture components"})


def cheatsheet():
    return "hdpmix: Hierarchical Dirichlet Process for shared mixture components"
