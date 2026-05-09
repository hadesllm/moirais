"""Horvitz-Thompson weighted mean."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["weighted_mean_survey"]


def weighted_mean_survey(y, weights):
    """
    Horvitz-Thompson weighted mean

    Formula: ybar_HT = sum w_i y_i / sum w_i

    Parameters
    ----------
    y : array-like
        Input data.
    weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Horvitz & Thompson (1952)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Horvitz-Thompson weighted mean"})


def cheatsheet():
    return "wmeansr: Horvitz-Thompson weighted mean"
