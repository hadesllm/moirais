"""Rényi entropy of order alpha."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["renyi_entropy"]


def renyi_entropy(p, alpha):
    """
    Rényi entropy of order alpha

    Formula: H_alpha = (1/(1-alpha)) log sum p^alpha

    Parameters
    ----------
    p : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rényi (1961)
    """
    p = np.atleast_1d(np.asarray(p, dtype=float))
    n = len(p)
    result = float(np.mean(p))
    se = float(np.std(p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rényi entropy of order alpha"})


def cheatsheet():
    return "reniyd: Rényi entropy of order alpha"
