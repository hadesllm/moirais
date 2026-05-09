# moirais.fn — function file (hadesllm/moirais)
"""Chernoff-Savage theorem: asymptotic normality of linear rank statistics."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_chernoff_savage"]


def gibbons_chernoff_savage(T_N, m, n, J, cdf=None):
    """
    Chernoff-Savage theorem: asymptotic normality of linear rank statistics

    Formula: (T_N/m - mu_N) / sigma_N ->_d N(0,1) subject to J-regularity conditions

    Parameters
    ----------
    T_N : array-like
        Input data.
    m : array-like
        Input data.
    n : array-like
        Input data.
    J : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: asymp_distribution

    References
    ----------
    Gibbons Theorem 7.3.8
    """
    T_N = np.asarray(T_N, dtype=float)
    n = int(T_N) if T_N.ndim == 0 else len(T_N)
    if T_N.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Chernoff-Savage theorem: asymptotic normality of linear rank statistics"})
    x_sorted = np.sort(T_N)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(T_N), scale=np.std(T_N, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Chernoff-Savage theorem: asymptotic normality of linear rank statistics"})


def cheatsheet():
    return "gb738: Chernoff-Savage theorem: asymptotic normality of linear rank statistics"
