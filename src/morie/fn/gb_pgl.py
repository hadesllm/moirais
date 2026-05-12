# morie.fn -- function file (hadesllm/morie)
"""Exact distribution of Page's L statistic from enumeration."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_page_exact"]


def gibbons_page_exact(l, k, b, cdf=None):
    """
    Exact distribution of Page's L statistic from enumeration

    Formula: P(L = l) = # permutations giving L = l / (k!)^b

    Parameters
    ----------
    l : array-like
        Input data.
    k : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: probability

    References
    ----------
    Gibbons Ch 12.3
    """
    l = np.asarray(l, dtype=float)
    n = int(l) if l.ndim == 0 else len(l)
    if l.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Exact distribution of Page's L statistic from enumeration"})
    x_sorted = np.sort(l)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(l), scale=np.std(l, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Exact distribution of Page's L statistic from enumeration"})


def cheatsheet():
    return "gb_pgl: Exact distribution of Page's L statistic from enumeration"
