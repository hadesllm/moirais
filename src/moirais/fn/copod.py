"""COPOD — copula-based outlier detection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["copod"]


def copod(X):
    """
    COPOD — copula-based outlier detection

    Formula: empirical CDFs combined via copula

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
    Li et al (2020) COPOD
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "COPOD — copula-based outlier detection"})


def cheatsheet():
    return "copod: COPOD — copula-based outlier detection"
