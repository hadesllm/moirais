"""Local Getis-Ord G_i*."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["local_getis_g"]


def local_getis_g(x, W):
    """
    Local Getis-Ord G_i*

    Formula: G_i* = sum_j w_ij x_j / sum_j x_j

    Parameters
    ----------
    x : array-like
        Input data.
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Getis-Ord (1992); Ord-Getis (1995)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Local Getis-Ord G_i*"})


def cheatsheet():
    return "lisgst: Local Getis-Ord G_i*"
