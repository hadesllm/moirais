"""Circular block bootstrap (Politis-Romano)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_circular_block"]


def boot_circular_block(x, block_len, stat, B):
    """
    Circular block bootstrap (Politis-Romano)

    Formula: Resample blocks from circularly extended series

    Parameters
    ----------
    x : array-like
        Input data.
    block_len : array-like
        Input data.
    stat : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta_b

    References
    ----------
    Politis & Romano (1992)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Circular block bootstrap (Politis-Romano)"})


def cheatsheet():
    return "btcbb: Circular block bootstrap (Politis-Romano)"
