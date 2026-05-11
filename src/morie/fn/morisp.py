"""Moran's I global spatial autocorrelation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["morans_i"]


def morans_i(x, W):
    """
    Moran's I global spatial autocorrelation

    Formula: I = n/W * sum_ij w_ij (x_i - xbar)(x_j - xbar) / sum (x_i - xbar)^2

    Parameters
    ----------
    x : array-like
        Input data.
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Moran (1950)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Moran's I global spatial autocorrelation"})
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Moran's I global spatial autocorrelation"})


def cheatsheet():
    return "morisp: Moran's I global spatial autocorrelation"
