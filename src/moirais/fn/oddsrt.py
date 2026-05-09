"""Odds ratio."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["odds_ratio"]


def odds_ratio(a, b, c, d):
    """
    Odds ratio

    Formula: OR = (a/b) / (c/d)

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    c : array-like
        Input data.
    d : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cornfield (1951)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Odds ratio"})


def cheatsheet():
    return "oddsrt: Odds ratio"
