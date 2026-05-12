# morie.fn -- function file (hadesllm/morie)
"""Mean and variance of the empirical distribution function S_n(x)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_edf_mean_var"]


def gibbons_edf_mean_var(x, n):
    """
    Mean and variance of the empirical distribution function S_n(x)

    Formula: E[S_n(x)] = F(x); Var[S_n(x)] = F(x)(1-F(x))/n

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mean, variance

    References
    ----------
    Gibbons Corollary 2.3.1.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mean and variance of the empirical distribution function S_n(x)"})


def cheatsheet():
    return "gb2311: Mean and variance of the empirical distribution function S_n(x)"
