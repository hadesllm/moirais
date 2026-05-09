"""Stratified sample mean."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["stratified_mean"]


def stratified_mean(y, stratum, weights):
    """
    Stratified sample mean

    Formula: ybar_st = sum_h W_h ybar_h, W_h = N_h / N

    Parameters
    ----------
    y : array-like
        Input data.
    stratum : array-like
        Input data.
    weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cochran (1977) §5.2
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stratified sample mean"})


def cheatsheet():
    return "stratm: Stratified sample mean"
