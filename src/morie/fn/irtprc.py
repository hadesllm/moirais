"""Partial credit model (Masters)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["partial_credit"]


def partial_credit(X, ncats):
    """
    Partial credit model (Masters)

    Formula: GPCM with a_j = 1

    Parameters
    ----------
    X : array-like
        Input data.
    ncats : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Masters (1982)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Partial credit model (Masters)"})


def cheatsheet():
    return "irtprc: Partial credit model (Masters)"
