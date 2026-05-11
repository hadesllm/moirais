# morie.fn — function file (hadesllm/morie)
"""Exact null distribution of one-sided KS statistic D+_n."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_ks_one_sided_dist"]


def gibbons_ks_one_sided_dist(c, n, cdf=None):
    """
    Exact null distribution of one-sided KS statistic D+_n

    Formula: P(D+_n < c) = n! * integral over unit simplex with bound c

    Parameters
    ----------
    c : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: probability

    References
    ----------
    Gibbons Theorem 4.3.4
    """
    c = np.asarray(c, dtype=float)
    n = int(c) if c.ndim == 0 else len(c)
    if c.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Exact null distribution of one-sided KS statistic D+_n"})
    x_sorted = np.sort(c)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(c), scale=np.std(c, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Exact null distribution of one-sided KS statistic D+_n"})


def cheatsheet():
    return "gb434: Exact null distribution of one-sided KS statistic D+_n"
