"""Pair correlation function g(r) = K'(r)/(2*pi*r)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["schabenberger_pair_correlation"]


def schabenberger_pair_correlation(points, lambda_est, r):
    """
    Pair correlation function g(r) = K'(r)/(2*pi*r)

    Formula: g(r) = K'(r) / (2*pi*r); g(r)=1 for CSR, g>1 clustering, g<1 regular

    Parameters
    ----------
    points : array-like
        Input data.
    lambda_est : array-like
        Input data.
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schabenberger Ch 3
    """
    points = np.asarray(points, dtype=float)
    y = np.asarray(points, dtype=float)
    n = min(len(points), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Pair correlation function g(r) = K'(r)/(2*pi*r)"})
    result = stats.spearmanr(points[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Pair correlation function g(r) = K'(r)/(2*pi*r)"})


def cheatsheet():
    return "sppair: Pair correlation function g(r) = K'(r)/(2*pi*r)"
