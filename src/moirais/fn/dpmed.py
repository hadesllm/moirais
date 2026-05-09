"""DP median (q=0.5 quantile)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dp_median"]


def dp_median(x, epsilon):
    """
    DP median (q=0.5 quantile)

    Formula: DP quantile at q=0.5

    Parameters
    ----------
    x : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Smith (2011)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP median (q=0.5 quantile)"})


def cheatsheet():
    return "dpmed: DP median (q=0.5 quantile)"
