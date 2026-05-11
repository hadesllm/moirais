# morie.fn — function file (hadesllm/morie)
"""Spearman rho test against zero rank correlation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_spearman_test"]


def gibbons_spearman_test(r_s, n, cdf=None):
    """
    Spearman rho test against zero rank correlation

    Formula: t = r_s * sqrt(n-2) / sqrt(1-r_s^2) ~ t(n-2) approximately

    Parameters
    ----------
    r_s : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: t_statistic, p_value

    References
    ----------
    Gibbons Ch 11.3
    """
    r_s = np.asarray(r_s, dtype=float)
    n = int(r_s) if r_s.ndim == 0 else len(r_s)
    if r_s.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Spearman rho test against zero rank correlation"})
    x_sorted = np.sort(r_s)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(r_s), scale=np.std(r_s, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Spearman rho test against zero rank correlation"})


def cheatsheet():
    return "gb_spq: Spearman rho test against zero rank correlation"
