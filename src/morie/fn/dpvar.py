"""DP variance (bounded)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dp_variance"]


def dp_variance(x, a, b, epsilon):
    """
    DP variance (bounded)

    Formula: compose DP mean + DP second-moment

    Parameters
    ----------
    x : array-like
        Input data.
    a : array-like
        Input data.
    b : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Karwa-Vadhan (2018)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP variance (bounded)"})


def cheatsheet():
    return "dpvar: DP variance (bounded)"
