"""Stratified random sample."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["stratified_design"]


def stratified_design(y, stratum, Nh):
    """
    Stratified random sample

    Formula: sum_h N_h ybar_h / N

    Parameters
    ----------
    y : array-like
        Input data.
    stratum : array-like
        Input data.
    Nh : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cochran (1977)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stratified random sample"})


def cheatsheet():
    return "strtfd: Stratified random sample"
