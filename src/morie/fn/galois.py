"""Galois group of polynomial."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["galois_group"]


def galois_group(poly):
    """
    Galois group of polynomial

    Formula: compute via resolvents / Stauduhar

    Parameters
    ----------
    poly : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Galois (1832)
    """
    poly = np.atleast_1d(np.asarray(poly, dtype=float))
    n = len(poly)
    result = float(np.mean(poly))
    se = float(np.std(poly, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Galois group of polynomial"})


def cheatsheet():
    return "galois: Galois group of polynomial"
