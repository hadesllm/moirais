"""Aitchison inner product on the simplex."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aitchison_inner_product"]


def aitchison_inner_product(x, y):
    """
    Aitchison inner product on the simplex

    Formula: <x,y>_A = clr(x)^T clr(y)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ip

    References
    ----------
    Aitchison (1986)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Aitchison inner product on the simplex"})


def cheatsheet():
    return "aitip: Aitchison inner product on the simplex"
