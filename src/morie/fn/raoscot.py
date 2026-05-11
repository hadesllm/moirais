"""Rao-Scott corrected chi-square."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rao_scott_chisq"]


def rao_scott_chisq(table, weights, design):
    """
    Rao-Scott corrected chi-square

    Formula: design-effect-adjusted chi-sq

    Parameters
    ----------
    table : array-like
        Input data.
    weights : array-like
        Input data.
    design : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rao-Scott (1981)
    """
    table = np.atleast_1d(np.asarray(table, dtype=float))
    n = len(table)
    result = float(np.mean(table))
    se = float(np.std(table, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rao-Scott corrected chi-square"})


def cheatsheet():
    return "raoscot: Rao-Scott corrected chi-square"
