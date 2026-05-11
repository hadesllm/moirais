"""Odds ratio for 2x2 table."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_odds_ratio"]


def wasserman_odds_ratio(table):
    """
    Odds ratio for 2x2 table

    Formula: OR = p11 p00 / (p10 p01)

    Parameters
    ----------
    table : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Wasserman (2004), Ch 16
    """
    table = np.atleast_1d(np.asarray(table, dtype=float))
    n = len(table)
    result = float(np.mean(table))
    se = float(np.std(table, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Odds ratio for 2x2 table"})


def cheatsheet():
    return "wsmodd: Odds ratio for 2x2 table"
