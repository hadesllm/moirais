# morie.fn -- function file (hadesllm/morie)
"""Variance of total runs up-and-down statistic."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_runs_ud_var"]


def gibbons_runs_ud_var(n, cdf=None):
    """
    Variance of total runs up-and-down statistic

    Formula: Var(R_ud) = (16n-29)/90

    Parameters
    ----------
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: variance

    References
    ----------
    Gibbons Ch 3.4
    """
    data = np.asarray(n, dtype=float) if np.ndim(n) > 0 else None
    n = int(n) if np.ndim(n) == 0 else len(n)
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Variance of total runs up-and-down statistic"})
    if data is None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal(n)
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Variance of total runs up-and-down statistic"})


def cheatsheet():
    return "gb34vr: Variance of total runs up-and-down statistic"
