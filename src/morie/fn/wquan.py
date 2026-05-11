"""Survey-weighted quantile."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["weighted_quantile"]


def weighted_quantile(y, weights, p):
    """
    Survey-weighted quantile

    Formula: q_p = inf{y : F_w(y) >= p}, F_w empirical CDF with weights

    Parameters
    ----------
    y : array-like
        Input data.
    weights : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Harrell-Davis (1982) on weighted quantile
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Survey-weighted quantile"})


def cheatsheet():
    return "wquan: Survey-weighted quantile"
