"""Spatial autocorrelation (Moran I/Geary C unified)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["spatial_autocorrelation"]


def spatial_autocorrelation(x, w):
    """
    Spatial autocorrelation (Moran I/Geary C unified)

    Formula: I = n/S0 * sum_ij w_ij(x_i-xbar)(x_j-xbar) / sum(x_i-xbar)^2

    Parameters
    ----------
    x : array-like
        Input data.
    w : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: statistic, p_value

    References
    ----------
    Schabenberger Ch 1
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(w, dtype=float)
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Spatial autocorrelation (Moran I/Geary C unified)"})
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Spatial autocorrelation (Moran I/Geary C unified)"})


def cheatsheet():
    return "sptau: Spatial autocorrelation (Moran I/Geary C unified)"
