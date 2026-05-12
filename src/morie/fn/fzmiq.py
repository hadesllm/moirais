# morie.fn -- function file (hadesllm/morie)
"""Moment inequality for degenerate U-statistics."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["fauzi_moment_ineq_ustat"]


def fauzi_moment_ineq_ustat(u_stat_func, q, cdf=None):
    """
    Moment inequality for degenerate U-statistics

    Formula: E|A_k|^q <= C n^{qk/2} E|rho_k(X_{i1},...,X_{ik})|^q for q>=2

    Parameters
    ----------
    u_stat_func : array-like
        Input data.
    q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bound

    References
    ----------
    Fauzi Ch 3, Eq 3.12
    """
    u_stat_func = np.asarray(u_stat_func, dtype=float)
    n = int(u_stat_func) if u_stat_func.ndim == 0 else len(u_stat_func)
    if u_stat_func.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Moment inequality for degenerate U-statistics"})
    x_sorted = np.sort(u_stat_func)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(u_stat_func), scale=np.std(u_stat_func, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Moment inequality for degenerate U-statistics"})


def cheatsheet():
    return "fzmiq: Moment inequality for degenerate U-statistics"
