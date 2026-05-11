# morie.fn — function file (hadesllm/morie)
"""Asymptotic normality of standardized total runs statistic."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_runs_asymp_normal"]


def gibbons_runs_asymp_normal(R, n, n1, n2, cdf=None):
    """
    Asymptotic normality of standardized total runs statistic

    Formula: Z = (R - 2nl(1-l)) / (2*sqrt(nl(1-l))) ->_d N(0,1)

    Parameters
    ----------
    R : array-like
        Input data.
    n : array-like
        Input data.
    n1 : array-like
        Input data.
    n2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: z_statistic

    References
    ----------
    Gibbons eq 3.2.9
    """
    R = np.asarray(R, dtype=float)
    n = int(R) if R.ndim == 0 else len(R)
    if R.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Asymptotic normality of standardized total runs statistic"})
    x_sorted = np.sort(R)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(R), scale=np.std(R, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Asymptotic normality of standardized total runs statistic"})


def cheatsheet():
    return "gb32asy: Asymptotic normality of standardized total runs statistic"
