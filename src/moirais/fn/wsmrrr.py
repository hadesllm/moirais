"""Relative risk for 2x2 table."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_relative_risk"]


def wasserman_relative_risk(table):
    """
    Relative risk for 2x2 table

    Formula: RR = p_{1|exposed} / p_{1|unexposed}

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Relative risk for 2x2 table"})


def cheatsheet():
    return "wsmrrr: Relative risk for 2x2 table"
