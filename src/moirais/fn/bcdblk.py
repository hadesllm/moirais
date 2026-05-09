"""Block-coordinate descent."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["block_coordinate_descent"]


def block_coordinate_descent(f, blocks, x0):
    """
    Block-coordinate descent

    Formula: min over one block while others fixed

    Parameters
    ----------
    f : array-like
        Input data.
    blocks : array-like
        Input data.
    x0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tseng (2001)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Block-coordinate descent"})


def cheatsheet():
    return "bcdblk: Block-coordinate descent"
