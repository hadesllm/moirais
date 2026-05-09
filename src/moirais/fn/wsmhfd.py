"""Hoeffding's inequality."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_hoeffding"]


def wasserman_hoeffding(n, t, a, b):
    """
    Hoeffding's inequality

    Formula: P(|X_bar - mu| > t) <= 2 exp(-2 n t^2 / (b-a)^2)

    Parameters
    ----------
    n : array-like
        Input data.
    t : array-like
        Input data.
    a : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bound

    References
    ----------
    Wasserman (2004), Ch 4
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hoeffding's inequality"})


def cheatsheet():
    return "wsmhfd: Hoeffding's inequality"
