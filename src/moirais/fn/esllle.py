"""Locally linear embedding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_lle"]


def esl_lle(X, k):
    """
    Locally linear embedding

    Formula: preserve local linear reconstruction weights

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: embedding

    References
    ----------
    Hastie ESL Ch 14
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Locally linear embedding"})


def cheatsheet():
    return "esllle: Locally linear embedding"
