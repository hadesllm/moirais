# morie.fn -- function file (hadesllm/morie)
"""Efficacy of two-sample t test for location under normal distribution."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_two_sample_t_efficacy"]


def gibbons_two_sample_t_efficacy(N, sigma, lam, cdf=None):
    """
    Efficacy of two-sample t test for location under normal distribution

    Formula: e(t) = N / (sigma^2 * lam(1-lam)) as n -> inf

    Parameters
    ----------
    N : array-like
        Input data.
    sigma : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: efficacy

    References
    ----------
    Gibbons Ch 13.3.2
    """
    N = np.asarray(N, dtype=float)
    n = int(N) if N.ndim == 0 else len(N)
    if N.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Efficacy of two-sample t test for location under normal distribution"})
    x_sorted = np.sort(N)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(N), scale=np.std(N, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Efficacy of two-sample t test for location under normal distribution"})


def cheatsheet():
    return "gb_ttm: Efficacy of two-sample t test for location under normal distribution"
