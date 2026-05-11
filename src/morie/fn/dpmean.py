"""DP mean (clamped + Laplace)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dp_mean"]


def dp_mean(x, a, b, epsilon):
    """
    DP mean (clamped + Laplace)

    Formula: M(D) = mean(clip(x,a,b)) + Lap((b−a)/(nε))

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
    Dwork-Roth (2014)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP mean (clamped + Laplace)"})


def cheatsheet():
    return "dpmean: DP mean (clamped + Laplace)"
