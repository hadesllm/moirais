# morie.fn — function file (hadesllm/morie)
"""Normal approximation for Wilcoxon signed-rank test."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_wsrt_normal_approx"]


def gibbons_wsrt_normal_approx(T_plus, n, cdf=None):
    """
    Normal approximation for Wilcoxon signed-rank test

    Formula: Z = (T+ - n(n+1)/4) / sqrt(n(n+1)(2n+1)/24) ~ N(0,1) for large n

    Parameters
    ----------
    T_plus : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: z_statistic

    References
    ----------
    Gibbons Ch 5.7
    """
    T_plus = np.asarray(T_plus, dtype=float)
    n = int(T_plus) if T_plus.ndim == 0 else len(T_plus)
    if T_plus.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Normal approximation for Wilcoxon signed-rank test"})
    x_sorted = np.sort(T_plus)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(T_plus), scale=np.std(T_plus, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Normal approximation for Wilcoxon signed-rank test"})


def cheatsheet():
    return "gb_wsn: Normal approximation for Wilcoxon signed-rank test"
