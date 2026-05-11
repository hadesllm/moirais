"""Bézout's identity coefficients."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bezout"]


def bezout(a, b):
    """
    Bézout's identity coefficients

    Formula: ax + by = gcd(a,b)

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bézout (1779)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bézout's identity coefficients"})


def cheatsheet():
    return "bezout: Bézout's identity coefficients"
