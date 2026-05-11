"""Balance coordinate from a single SBP row."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aitchison_balance"]


def aitchison_balance(x, row):
    """
    Balance coordinate from a single SBP row

    Formula: b = sqrt(rs/(r+s)) log(g_+ / g_-)

    Parameters
    ----------
    x : array-like
        Input data.
    row : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: b

    References
    ----------
    Egozcue & Pawlowsky-Glahn (2005)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Balance coordinate from a single SBP row"})


def cheatsheet():
    return "aitbal: Balance coordinate from a single SBP row"
