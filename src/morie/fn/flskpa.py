"""Fleiss kappa for multiple raters."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["fleiss_kappa"]


def fleiss_kappa(X):
    """
    Fleiss kappa for multiple raters

    Formula: avg agreement adjusted for chance

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fleiss (1971)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fleiss kappa for multiple raters"})


def cheatsheet():
    return "flskpa: Fleiss kappa for multiple raters"
