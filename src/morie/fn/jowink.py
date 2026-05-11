# morie.fn — function file (hadesllm/morie)
"""Winkler interval score: penalize narrow intervals + miscoverage."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_winkler_interval_score"]


def joseph_winkler_interval_score(y, l, u, alpha):
    """
    Winkler interval score: penalize narrow intervals + miscoverage

    Formula: W = (u - l) + (2/alpha)*(l - y)*1{y<l} + (2/alpha)*(y - u)*1{y>u}

    Parameters
    ----------
    y : array-like
        Input data.
    l : array-like
        Input data.
    u : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: score

    References
    ----------
    Joseph Ch 17, Interval Score (Winkler) section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Winkler interval score: penalize narrow intervals + miscoverage"})


def cheatsheet():
    return "jowink: Winkler interval score: penalize narrow intervals + miscoverage"
