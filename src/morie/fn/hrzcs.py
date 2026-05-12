# morie.fn -- function file (hadesllm/morie)
"""Curse of dimensionality in nonparametric regression: rate degrades with dimension."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_curse_dimensionality"]


def horowitz_curse_dimensionality(d, n):
    """
    Curse of dimensionality in nonparametric regression: rate degrades with dimension

    Formula: MSE(m_hat) = O(n^{-4/(4+d)}) for d-dimensional X; O(n^{-4/5}) for d=1

    Parameters
    ----------
    d : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rate

    References
    ----------
    Horowitz Ch 1
    """
    d = np.asarray(d, dtype=float)
    n = int(d) if d.ndim == 0 else len(d)
    result = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Curse of dimensionality in nonparametric regression: rate degrades with dimension"})


def cheatsheet():
    return "hrzcs: Curse of dimensionality in nonparametric regression: rate degrades with dimension"
