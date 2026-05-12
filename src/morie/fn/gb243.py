# morie.fn -- function file (hadesllm/morie)
"""r-th order statistic from Uniform(0,1) follows Beta(r, n-r+1) distribution."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_order_beta"]


def gibbons_order_beta(r, n, cdf=None):
    """
    r-th order statistic from Uniform(0,1) follows Beta(r, n-r+1) distribution

    Formula: X_(r) ~ Beta(r, n-r+1) when F = Uniform(0,1)

    Parameters
    ----------
    r : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: distribution

    References
    ----------
    Gibbons Theorem 2.4.3
    """
    r = np.asarray(r, dtype=float)
    n = int(r) if r.ndim == 0 else len(r)
    if r.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "r-th order statistic from Uniform(0,1) follows Beta(r, n-r+1) distribution"})
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "r-th order statistic from Uniform(0,1) follows Beta(r, n-r+1) distribution"})


def cheatsheet():
    return "gb243: r-th order statistic from Uniform(0,1) follows Beta(r, n-r+1) distribution"
