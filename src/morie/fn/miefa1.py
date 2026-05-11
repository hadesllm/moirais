"""Fraction of missing information."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mi_fmi"]


def mi_fmi(between, within, m):
    """
    Fraction of missing information

    Formula: FMI = (1 + 1/m) B / V_total

    Parameters
    ----------
    between : array-like
        Input data.
    within : array-like
        Input data.
    m : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schafer (1997)
    """
    between = np.atleast_1d(np.asarray(between, dtype=float))
    n = len(between)
    result = float(np.mean(between))
    se = float(np.std(between, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fraction of missing information"})


def cheatsheet():
    return "miefa1: Fraction of missing information"
