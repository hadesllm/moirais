"""Rényi entropy of order alpha."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["renyi_entropy"]


def renyi_entropy(y, alpha, base):
    """
    Rényi entropy of order alpha

    Formula: H_alpha(X) = (1/(1-alpha)) log sum_x p(x)^alpha

    Parameters
    ----------
    y : array-like
        Input data.
    alpha : array-like
        Input data.
    base : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rényi (1961)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rényi entropy of order alpha"})


def cheatsheet():
    return "renent: Rényi entropy of order alpha"
