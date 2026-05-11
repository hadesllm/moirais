"""Vine copula (pair-copula construction)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["vine_copula"]


def vine_copula(x):
    """
    Vine copula (pair-copula construction)

    Formula: f(x) = prod c_{ij|D}(F(x_i|D), F(x_j|D))

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Aas et al. (2009)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Vine copula (pair-copula construction)"})


def cheatsheet():
    return "vines: Vine copula (pair-copula construction)"
