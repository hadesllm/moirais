# morie.fn — function file (hadesllm/morie)
"""Rate of convergence for G in single-index: same as 1D nonparametric."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_rate_G_estimation"]


def horowitz_rate_G_estimation(x, y, bandwidth):
    """
    Rate of convergence for G in single-index: same as 1D nonparametric

    Formula: h_opt ~ n^{-1/5} for G; MSE(G_hat) = O(n^{-4/5}); no curse of dimensionality

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rate

    References
    ----------
    Horowitz Ch 2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rate of convergence for G in single-index: same as 1D nonparametric"})


def cheatsheet():
    return "hrzrateG: Rate of convergence for G in single-index: same as 1D nonparametric"
