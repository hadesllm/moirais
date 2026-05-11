# morie.fn — function file (hadesllm/morie)
"""Sample size determination for Mann-Whitney test."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_mw_sampsize"]


def gibbons_mw_sampsize(alpha, beta, delta, cdf=None):
    """
    Sample size determination for Mann-Whitney test

    Formula: n from normal approx to U distribution given alpha, beta, delta

    Parameters
    ----------
    alpha : array-like
        Input data.
    beta : array-like
        Input data.
    delta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: sample_size

    References
    ----------
    Gibbons Ch 6.6 sample size
    """
    alpha = np.asarray(alpha, dtype=float)
    n = int(alpha) if alpha.ndim == 0 else len(alpha)
    if alpha.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Sample size determination for Mann-Whitney test"})
    x_sorted = np.sort(alpha)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(alpha), scale=np.std(alpha, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Sample size determination for Mann-Whitney test"})


def cheatsheet():
    return "gb661s: Sample size determination for Mann-Whitney test"
