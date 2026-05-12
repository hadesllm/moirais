# morie.fn -- function file (hadesllm/morie)
"""Covariance of two linear rank statistics B_N and T_N."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_linrank_covariance"]


def gibbons_linrank_covariance(a, b, m, n, N, cdf=None):
    """
    Covariance of two linear rank statistics B_N and T_N

    Formula: cov(B,T) = mn/(N^2(N-1)) * (N*sum(a_i b_i) - sum(a_i)*sum(b_i))

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    m : array-like
        Input data.
    n : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: covariance

    References
    ----------
    Gibbons Theorem 7.3.3
    """
    a = np.asarray(a, dtype=float)
    n = int(a) if a.ndim == 0 else len(a)
    if a.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Covariance of two linear rank statistics B_N and T_N"})
    x_sorted = np.sort(a)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(a), scale=np.std(a, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Covariance of two linear rank statistics B_N and T_N"})


def cheatsheet():
    return "gb733: Covariance of two linear rank statistics B_N and T_N"
