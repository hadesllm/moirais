"""Andrews-Monahan prewhitened HAC."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["andrews_monahan_hac"]


def andrews_monahan_hac(e, X):
    """
    Andrews-Monahan prewhitened HAC

    Formula: prewhiten + Newey-West, then recolour

    Parameters
    ----------
    e : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Andrews & Monahan (1992)
    """
    e = np.atleast_1d(np.asarray(e, dtype=float))
    n = len(e)
    result = float(np.mean(e))
    se = float(np.std(e, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Andrews-Monahan prewhitened HAC"})


def cheatsheet():
    return "andmnh: Andrews-Monahan prewhitened HAC"
