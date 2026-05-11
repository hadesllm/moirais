# morie.fn — function file (hadesllm/morie)
"""Randomized test procedure when exact size is unavailable at a discrete level."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_rz_test"]


def gibbons_rz_test(statistic, null_dist, alpha, cdf=None):
    """
    Randomized test procedure when exact size is unavailable at a discrete level

    Formula: Randomize with probability gamma = (alpha - P1)/(P2 - P1) at boundary

    Parameters
    ----------
    statistic : array-like
        Input data.
    null_dist : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: decision

    References
    ----------
    Gibbons Ch 1.2.12
    """
    statistic = np.asarray(statistic, dtype=float)
    n = int(statistic) if statistic.ndim == 0 else len(statistic)
    if statistic.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Randomized test procedure when exact size is unavailable at a discrete level"})
    x_sorted = np.sort(statistic)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(statistic), scale=np.std(statistic, ddof=1))
    else:
        cdf_vals = np.array([cdf(xi) for xi in x_sorted])
    ecdf = np.arange(1, n + 1) / n
    ecdf_prev = np.arange(0, n) / n
    d_plus = np.max(ecdf - cdf_vals)
    d_minus = np.max(cdf_vals - ecdf_prev)
    statistic = max(d_plus, d_minus)
    if n <= 40:
        p_value = 1.0 - stats.ksone.cdf(statistic, n)
    else:
        lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * statistic
        p_value = 2.0 * np.sum([(-1) ** (k - 1) * np.exp(-2 * k ** 2 * lam ** 2) for k in range(1, 101)])
        p_value = max(0.0, min(1.0, p_value))
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Randomized test procedure when exact size is unavailable at a discrete level"})


def cheatsheet():
    return "gb_rz: Randomized test procedure when exact size is unavailable at a discrete level"
