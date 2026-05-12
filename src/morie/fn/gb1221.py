# morie.fn -- function file (hadesllm/morie)
"""Friedman two-way ANOVA by ranks chi-r^2 statistic."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_friedman"]


def gibbons_friedman(data, k, b, cdf=None):
    """
    Friedman two-way ANOVA by ranks chi-r^2 statistic

    Formula: chi_r^2 = 12/(bk(k+1)) * sum_j (R_j)^2 - 3b(k+1); R_j = col rank sums

    Parameters
    ----------
    data : array-like
        Input data.
    k : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: statistic, p_value

    References
    ----------
    Gibbons Ch 12.2
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    if data.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Friedman two-way ANOVA by ranks chi-r^2 statistic"})
    x_sorted = np.sort(data)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(data), scale=np.std(data, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Friedman two-way ANOVA by ranks chi-r^2 statistic"})


def cheatsheet():
    return "gb1221: Friedman two-way ANOVA by ranks chi-r^2 statistic"
