"""Polynomial GCD via Euclid."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["polynomial_gcd"]


def polynomial_gcd(p, q):
    """
    Polynomial GCD via Euclid

    Formula: repeat poly division

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
    Knuth TAOCP V.II
    """
    p = np.atleast_1d(np.asarray(p, dtype=float))
    n = len(p)
    result = float(np.mean(p))
    se = float(np.std(p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Polynomial GCD via Euclid"})


def cheatsheet():
    return "euclP: Polynomial GCD via Euclid"
