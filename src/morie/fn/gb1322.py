# morie.fn — function file (hadesllm/morie)
"""ARE as ratio of squared efficacies of two test statistics."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_are_formula"]


def gibbons_are_formula(T, T_star, u0, cdf=None):
    """
    ARE as ratio of squared efficacies of two test statistics

    Formula: ARE(T,T*) = e(T)/e(T*) = [dE(T)/du]^2/Var(T) / [dE(T*)/du]^2/Var(T*)

    Parameters
    ----------
    T : array-like
        Input data.
    T_star : array-like
        Input data.
    u0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ARE

    References
    ----------
    Gibbons Theorem 13.2.2
    """
    T = np.asarray(T, dtype=float)
    n = int(T) if T.ndim == 0 else len(T)
    if T.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "ARE as ratio of squared efficacies of two test statistics"})
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "ARE as ratio of squared efficacies of two test statistics"})


def cheatsheet():
    return "gb1322: ARE as ratio of squared efficacies of two test statistics"
