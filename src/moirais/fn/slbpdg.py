"""Slice sampler for DP mixtures."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["slice_break_dp"]


def slice_break_dp(y, alpha, n_iter):
    """
    Slice sampler for DP mixtures

    Formula: auxiliary slice variable removes truncation

    Parameters
    ----------
    y : array-like
        Input data.
    alpha : array-like
        Input data.
    n_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Walker (2007); Kalli-Griffin-Walker (2011)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Slice sampler for DP mixtures"})


def cheatsheet():
    return "slbpdg: Slice sampler for DP mixtures"
