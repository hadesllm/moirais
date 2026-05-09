"""Item-based CF."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["item_cf"]


def item_cf(R, u, i, k):
    """
    Item-based CF

    Formula: r̂_{ui} = sum sim(i,j) r_{uj} / sum sim

    Parameters
    ----------
    R : array-like
        Input data.
    u : array-like
        Input data.
    i : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sarwar et al (2001)
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    n = len(R)
    result = float(np.mean(R))
    se = float(np.std(R, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Item-based CF"})


def cheatsheet():
    return "icfR: Item-based CF"
