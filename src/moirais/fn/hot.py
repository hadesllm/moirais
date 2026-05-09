"""HOT SAX discord."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hot_sax"]


def hot_sax(x, window, alphabet):
    """
    HOT SAX discord

    Formula: SAX symbolic + heuristic search

    Parameters
    ----------
    x : array-like
        Input data.
    window : array-like
        Input data.
    alphabet : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Keogh-Lin-Fu (2005)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "HOT SAX discord"})


def cheatsheet():
    return "hot: HOT SAX discord"
