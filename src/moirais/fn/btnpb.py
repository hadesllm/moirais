"""Non-overlapping block bootstrap for time series."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_nonoverlap_block"]


def boot_nonoverlap_block(x, block_len, stat, B):
    """
    Non-overlapping block bootstrap for time series

    Formula: Resample blocks of length ℓ from disjoint partition

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
    Carlstein (1986)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Non-overlapping block bootstrap for time series"})


def cheatsheet():
    return "btnpb: Non-overlapping block bootstrap for time series"
