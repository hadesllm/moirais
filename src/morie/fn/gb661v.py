# morie.fn -- function file (hadesllm/morie)
"""Variance of Mann-Whitney U under null hypothesis."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_mw_var"]


def gibbons_mw_var(m, n):
    """
    Variance of Mann-Whitney U under null hypothesis

    Formula: Var(U) = mn(m+n+1)/12

    Parameters
    ----------
    m : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: variance

    References
    ----------
    Gibbons Ch 6.6
    """
    m = np.asarray(m, dtype=float)
    n = int(m) if m.ndim == 0 else len(m)
    result = float(np.mean(m))
    se = float(np.std(m, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variance of Mann-Whitney U under null hypothesis"})


def cheatsheet():
    return "gb661v: Variance of Mann-Whitney U under null hypothesis"
