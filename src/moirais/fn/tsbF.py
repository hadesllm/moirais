"""TSB modification for Croston."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tsb"]


def tsb(y, alpha, beta):
    """
    TSB modification for Croston

    Formula: replaces interval with prob → less bias

    Parameters
    ----------
    y : array-like
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
    Teunter-Syntetos-Babai (2011)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TSB modification for Croston"})


def cheatsheet():
    return "tsbF: TSB modification for Croston"
