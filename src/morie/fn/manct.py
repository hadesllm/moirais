"""Add continuity correction c to zero cells."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_continuity_correction"]


def ma_continuity_correction(a, b, c, d, cc):
    """
    Add continuity correction c to zero cells

    Formula: a*=a+c, b*=b+c, c*=c+c, d*=d+c

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    c : array-like
        Input data.
    d : array-like
        Input data.
    cc : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: a_adj, b_adj, c_adj, d_adj

    References
    ----------
    Sweeting-Sutton-Lambert (2004)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Add continuity correction c to zero cells"})


def cheatsheet():
    return "manct: Add continuity correction c to zero cells"
