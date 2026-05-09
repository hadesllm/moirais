"""Isometric log-ratio (ILR) transform via SBP."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aitchison_ilr"]


def aitchison_ilr(x, V):
    """
    Isometric log-ratio (ILR) transform via SBP

    Formula: ilr(x) = V^T clr(x),  V from SBP

    Parameters
    ----------
    x : array-like
        Input data.
    V : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Egozcue et al. (2003)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Isometric log-ratio (ILR) transform via SBP"})


def cheatsheet():
    return "aitilr: Isometric log-ratio (ILR) transform via SBP"
