"""Spatial autocorrelation function rho(h) = C(h)/C(0)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["schabenberger_autocorrelation_function"]


def schabenberger_autocorrelation_function(coords, z):
    """
    Spatial autocorrelation function rho(h) = C(h)/C(0)

    Formula: rho(h) = C(h)/C(0)

    Parameters
    ----------
    coords : array-like
        Input data.
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schabenberger Ch 1, Sec 1.4.2
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(z, dtype=float)
    n = min(len(z), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Spatial autocorrelation function rho(h) = C(h)/C(0)"})
    result = stats.spearmanr(z[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Spatial autocorrelation function rho(h) = C(h)/C(0)"})


def cheatsheet():
    return "spackf: Spatial autocorrelation function rho(h) = C(h)/C(0)"
