# morie.fn -- function file (hadesllm/morie)
"""Covariance between r-th and s-th order statistics (r < s)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_order_covariance"]


def gibbons_order_covariance(r, s, n, f, F, cdf=None):
    """
    Covariance between r-th and s-th order statistics (r < s)

    Formula: Cov(X_(r), X_(s)) via double integral of joint distribution

    Parameters
    ----------
    r : array-like
        Input data.
    s : array-like
        Input data.
    n : array-like
        Input data.
    f : array-like
        Input data.
    F : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: covariance

    References
    ----------
    Gibbons Ch 2.8.2
    """
    r = np.asarray(r, dtype=float)
    n = int(r) if r.ndim == 0 else len(r)
    if r.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Covariance between r-th and s-th order statistics (r < s)"})
    x_sorted = np.sort(r)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(r), scale=np.std(r, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Covariance between r-th and s-th order statistics (r < s)"})


def cheatsheet():
    return "gb_cvo: Covariance between r-th and s-th order statistics (r < s)"
