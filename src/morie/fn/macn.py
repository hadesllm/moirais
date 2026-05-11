"""Cochran's Q statistic for heterogeneity."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_cochran_q"]


def ma_cochran_q(yi, vi, cdf=None):
    """
    Cochran's Q statistic for heterogeneity

    Formula: Q = Σ w_i (y_i - θ̂_FE)²

    Parameters
    ----------
    yi : array-like
        Input data.
    vi : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Q, df, p

    References
    ----------
    Cochran (1954)
    """
    yi = np.asarray(yi, dtype=float)
    n = len(yi)
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Cochran's Q statistic for heterogeneity"})
    x_sorted = np.sort(yi)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(yi), scale=np.std(yi, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Cochran's Q statistic for heterogeneity"})


def cheatsheet():
    return "macn: Cochran's Q statistic for heterogeneity"
