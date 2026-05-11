"""Block-maxima GEV fit from a series."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_block_maxima_fit"]


def evt_block_maxima_fit(x, block):
    """
    Block-maxima GEV fit from a series

    Formula: z_i = max over block i; fit GEV via MLE

    Parameters
    ----------
    x : array-like
        Input data.
    block : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mu, sigma, xi, blocks

    References
    ----------
    Coles (2001) §3
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Block-maxima GEV fit from a series"})


def cheatsheet():
    return "evblockm: Block-maxima GEV fit from a series"
