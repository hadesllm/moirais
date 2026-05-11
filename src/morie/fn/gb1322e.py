# morie.fn — function file (hadesllm/morie)
"""Efficacy e(T_n) of test statistic T_n."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_efficacy"]


def gibbons_efficacy(T, u0, cdf=None):
    """
    Efficacy e(T_n) of test statistic T_n

    Formula: e(T_n) = [dE(T_n)/du|_{u=u0}]^2 / Var(T_n)|_{u=u0}

    Parameters
    ----------
    T : array-like
        Input data.
    u0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: efficacy

    References
    ----------
    Gibbons eq 13.2.4
    """
    T = np.asarray(T, dtype=float)
    n = int(T) if T.ndim == 0 else len(T)
    if T.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Efficacy e(T_n) of test statistic T_n"})
    x_sorted = np.sort(T)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(T), scale=np.std(T, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Efficacy e(T_n) of test statistic T_n"})


def cheatsheet():
    return "gb1322e: Efficacy e(T_n) of test statistic T_n"
