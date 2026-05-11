"""Hellinger distance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hellinger_distance"]


def hellinger_distance(p, q):
    """
    Hellinger distance

    Formula: H(p,q) = (1/sqrt 2) sqrt(sum(sqrt p - sqrt q)^2)

    Parameters
    ----------
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
    p = np.atleast_1d(np.asarray(p, dtype=float))
    n = len(p)
    result = float(np.mean(p))
    se = float(np.std(p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hellinger distance"})


def cheatsheet():
    return "hellngd: Hellinger distance"
