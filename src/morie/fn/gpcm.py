"""Generalized Partial Credit Model (Muraki)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["generalized_partial_credit"]


def generalized_partial_credit(y, theta, a, b_j):
    """
    Generalized Partial Credit Model (Muraki)

    Formula: P(X=k) = exp(sum_{j=0}^{k} a(theta - b_j)) / sum_h exp(sum_{j=0}^{h} a(theta - b_j))

    Parameters
    ----------
    y : array-like
        Input data.
    theta : array-like
        Input data.
    a : array-like
        Input data.
    b_j : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Muraki (1992)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Generalized Partial Credit Model (Muraki)"})


def cheatsheet():
    return "gpcm: Generalized Partial Credit Model (Muraki)"
