# morie.fn -- function file (hadesllm/morie)
"""Efficacy of sign test for location parameter."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_sign_efficacy"]


def gibbons_sign_efficacy(N, f, median, cdf=None):
    """
    Efficacy of sign test for location parameter

    Formula: e(K_N) = 4*N*f^2(median) = 4*N*f^2(F^{-1}(0.5))

    Parameters
    ----------
    N : array-like
        Input data.
    f : array-like
        Input data.
    median : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: efficacy

    References
    ----------
    Gibbons eq 13.3.3
    """
    N = np.asarray(N, dtype=float)
    n = int(N) if N.ndim == 0 else len(N)
    if N.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Efficacy of sign test for location parameter"})
    x_sorted = np.sort(N)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(N), scale=np.std(N, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Efficacy of sign test for location parameter"})


def cheatsheet():
    return "gb1331s: Efficacy of sign test for location parameter"
