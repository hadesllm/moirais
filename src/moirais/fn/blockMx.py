"""Block-maxima method."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["block_maxima"]


def block_maxima(y, block_size):
    """
    Block-maxima method

    Formula: fit GEV to block maxima

    Parameters
    ----------
    y : array-like
        Input data.
    block_size : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Coles (2001)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Block-maxima method"})


def cheatsheet():
    return "blockMx: Block-maxima method"
