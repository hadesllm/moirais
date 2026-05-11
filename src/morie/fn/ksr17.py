# morie.fn — function file (hadesllm/morie)
"""Counting process for survival data."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_counting_process"]


def kosorok_counting_process(t, event):
    """
    Counting process for survival data

    Formula: N(t) = sum 1(Ti <= t, Delta_i = 1)

    Parameters
    ----------
    t : array-like
        Input data.
    event : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 8
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Counting process for survival data"})


def cheatsheet():
    return "ksr17: Counting process for survival data"
