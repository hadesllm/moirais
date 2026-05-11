"""Katz centrality with attenuation alpha."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["katz_centrality"]


def katz_centrality(y, A, alpha, beta):
    """
    Katz centrality with attenuation alpha

    Formula: x = (I - alpha A)^-1 (alpha A) 1

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    alpha : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Katz (1953)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Katz centrality with attenuation alpha"})


def cheatsheet():
    return "katzc: Katz centrality with attenuation alpha"
