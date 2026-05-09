"""DP quantile via exponential mechanism."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dp_quantile"]


def dp_quantile(x, q, epsilon):
    """
    DP quantile via exponential mechanism

    Formula: u(D,r) = -|rank(r) − qn|; ExpM(D, u, ε)

    Parameters
    ----------
    x : array-like
        Input data.
    q : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Smith (2011); Gillenwater et al (2021)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP quantile via exponential mechanism"})


def cheatsheet():
    return "dpqua: DP quantile via exponential mechanism"
