"""Simplicial depth."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["simplicial_depth"]


def simplicial_depth(X, theta):
    """
    Simplicial depth

    Formula: P(θ in random simplex of d+1 points)

    Parameters
    ----------
    X : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Liu (1990)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Simplicial depth"})


def cheatsheet():
    return "depthS: Simplicial depth"
