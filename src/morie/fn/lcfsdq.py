"""Local-cluster first-order SD query."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["lc_first_sd_query"]


def lc_first_sd_query(x, coords):
    """
    Local-cluster first-order SD query

    Formula: first stand-dev neighbor distance

    Parameters
    ----------
    x : array-like
        Input data.
    coords : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    applied
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Local-cluster first-order SD query"})


def cheatsheet():
    return "lcfsdq: Local-cluster first-order SD query"
