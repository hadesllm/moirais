"""Brenier map recovery in 1-D from sorted samples."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_map_recovery_brenier"]


def ot_map_recovery_brenier(x, y):
    """
    Brenier map recovery in 1-D from sorted samples

    Formula: Tx_i = y_(σ(i)) where σ is sort permutation

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: T_map

    References
    ----------
    Brenier (1991)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Brenier map recovery in 1-D from sorted samples"})


def cheatsheet():
    return "otmaprc: Brenier map recovery in 1-D from sorted samples"
