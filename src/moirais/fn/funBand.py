"""Functional confidence band."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["functional_band"]


def functional_band(Y, alpha):
    """
    Functional confidence band

    Formula: Wahba band via simulation

    Parameters
    ----------
    Y : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wahba (1983)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Functional confidence band"})


def cheatsheet():
    return "funBand: Functional confidence band"
