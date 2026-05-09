"""Weak law of large numbers."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_lln"]


def wasserman_lln(data):
    """
    Weak law of large numbers

    Formula: X_bar_n -> mu in probability

    Parameters
    ----------
    data : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wasserman (2004), Ch 5
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Weak law of large numbers"})


def cheatsheet():
    return "wsmlln: Weak law of large numbers"
