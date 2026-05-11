"""Kolmogorov-Smirnov test."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_ks_test"]


def wasserman_ks_test(data, f0, cdf=None):
    """
    Kolmogorov-Smirnov test

    Formula: D = sup_x |F_n(x) - F_0(x)|

    Parameters
    ----------
    data : array-like
        Input data.
    f0 : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: statistic, p_value

    References
    ----------
    Wasserman (2004), Ch 10
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Kolmogorov-Smirnov test"})
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Kolmogorov-Smirnov test"})


def cheatsheet():
    return "wsmksm: Kolmogorov-Smirnov test"
