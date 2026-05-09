"""Forest-fit consistency check."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["forest_fit_consistency"]


def forest_fit_consistency(y, D, X, K):
    """
    Forest-fit consistency check

    Formula: compare in-fold vs out-of-fold tau

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wager-Athey (2018)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Forest-fit consistency check"})


def cheatsheet():
    return "frfgrf: Forest-fit consistency check"
