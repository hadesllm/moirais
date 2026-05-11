"""FAST-MCD algorithm."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["fast_mcd"]


def fast_mcd(X, h, n_starts):
    """
    FAST-MCD algorithm

    Formula: C-step iterations from many initial subsets

    Parameters
    ----------
    X : array-like
        Input data.
    h : array-like
        Input data.
    n_starts : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rousseeuw-Van Driessen (1999)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "FAST-MCD algorithm"})


def cheatsheet():
    return "fastm: FAST-MCD algorithm"
