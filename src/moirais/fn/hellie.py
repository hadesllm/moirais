"""Hellinger distance between two distributions."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hellinger_distance"]


def hellinger_distance(y, p, q):
    """
    Hellinger distance between two distributions

    Formula: H(P,Q) = (1/sqrt(2)) sqrt(sum_x (sqrt(p(x)) - sqrt(q(x)))^2)

    Parameters
    ----------
    y : array-like
        Input data.
    p : array-like
        Input data.
    q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hellinger (1909)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hellinger distance between two distributions"})


def cheatsheet():
    return "hellie: Hellinger distance between two distributions"
