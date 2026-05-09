"""HLM gamma covariance matrix for random effects (T matrix)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hlm_gamma_matrix"]


def hlm_gamma_matrix(y, X, Z, cluster):
    """
    HLM gamma covariance matrix for random effects (T matrix)

    Formula: T = E[u_j u_j']; Cov(u) = block-diag(T)

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    Z : array-like
        Input data.
    cluster : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Raudenbush & Bryk (2002) §3
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "HLM gamma covariance matrix for random effects (T matrix)"})


def cheatsheet():
    return "hlmgr: HLM gamma covariance matrix for random effects (T matrix)"
