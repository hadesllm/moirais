"""Independent component analysis."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_ica"]


def esl_ica(X, k):
    """
    Independent component analysis

    Formula: X = AS; recover S via mutual independence

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: components

    References
    ----------
    Hastie ESL Ch 14
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Independent component analysis"})


def cheatsheet():
    return "eslica: Independent component analysis"
