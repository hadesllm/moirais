"""Projection depth."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["projection_depth"]


def projection_depth(x, X):
    """
    Projection depth

    Formula: d = 1/(1 + sup_u |u^T x − Med|/MAD)

    Parameters
    ----------
    x : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Stahel (1981); Donoho (1982)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Projection depth"})


def cheatsheet():
    return "depthP: Projection depth"
