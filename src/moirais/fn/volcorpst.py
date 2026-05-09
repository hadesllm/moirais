"""Corradi-Swanson long-horizon predictability test."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_corradi_swan_persistence"]


def vol_corradi_swan_persistence(r, horizons, cdf=None):
    """
    Corradi-Swanson long-horizon predictability test

    Formula: Test long-run vol predictive power vs benchmark

    Parameters
    ----------
    r : array-like
        Input data.
    horizons : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: stat, p

    References
    ----------
    Corradi-Swanson (2006)
    """
    r = np.asarray(r, dtype=float)
    n = len(r)
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Corradi-Swanson long-horizon predictability test"})
    x_sorted = np.sort(r)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(r), scale=np.std(r, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Corradi-Swanson long-horizon predictability test"})


def cheatsheet():
    return "volcorpst: Corradi-Swanson long-horizon predictability test"
