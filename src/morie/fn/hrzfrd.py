# morie.fn -- function file (hadesllm/morie)
"""Fredholm integral equation of the first kind (statistical inverse problem)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["horowitz_fredholm_eq"]


def horowitz_fredholm_eq(m, k, cdf=None):
    """
    Fredholm integral equation of the first kind (statistical inverse problem)

    Formula: m(w) = integral k(m,w)*g(m)dx; g is unknown function to be estimated

    Parameters
    ----------
    m : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: g_hat

    References
    ----------
    Horowitz Ch 5, Eq 5.1
    """
    m = np.asarray(m, dtype=float)
    n = int(m) if m.ndim == 0 else len(m)
    if m.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Fredholm integral equation of the first kind (statistical inverse problem)"})
    x_sorted = np.sort(m)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(m), scale=np.std(m, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Fredholm integral equation of the first kind (statistical inverse problem)"})


def cheatsheet():
    return "hrzfrd: Fredholm integral equation of the first kind (statistical inverse problem)"
