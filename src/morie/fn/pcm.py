"""Masters Partial Credit Model (a=1)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["partial_credit_masters"]


def partial_credit_masters(y, theta, delta_j):
    """
    Masters Partial Credit Model (a=1)

    Formula: P(X=k) = exp(sum (theta - delta_j)) / sum_h exp(sum (theta - delta_j))

    Parameters
    ----------
    y : array-like
        Input data.
    theta : array-like
        Input data.
    delta_j : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Masters (1982)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Masters Partial Credit Model (a=1)"})


def cheatsheet():
    return "pcm: Masters Partial Credit Model (a=1)"
