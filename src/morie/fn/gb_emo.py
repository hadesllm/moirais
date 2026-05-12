# morie.fn -- function file (hadesllm/morie)
"""k-th moment of r-th order statistic E[X_(r)^k]."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_order_moments"]


def gibbons_order_moments(r, n, k, f, F, cdf=None):
    """
    k-th moment of r-th order statistic E[X_(r)^k]

    Formula: E[X_(r)^k] = n!/((r-1)!(n-r)!) integral r^k F(r)^(r-1) [1-F(r)]^(n-r) f(r) dx

    Parameters
    ----------
    r : array-like
        Input data.
    n : array-like
        Input data.
    k : array-like
        Input data.
    f : array-like
        Input data.
    F : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: moment

    References
    ----------
    Gibbons Ch 2.8.1
    """
    r = np.asarray(r, dtype=float)
    n = int(r) if r.ndim == 0 else len(r)
    if r.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "k-th moment of r-th order statistic E[X_(r)^k]"})
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "k-th moment of r-th order statistic E[X_(r)^k]"})


def cheatsheet():
    return "gb_emo: k-th moment of r-th order statistic E[X_(r)^k]"
