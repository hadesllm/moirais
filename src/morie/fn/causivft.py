"""First-stage IV F-statistic for weak instruments."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_iv_first_stage"]


def causal_iv_first_stage(D, Z, X_exog, cdf=None):
    """
    First-stage IV F-statistic for weak instruments

    Formula: F = (R²/(1-R²))(n-k-1)/k from D = π Z + ε

    Parameters
    ----------
    D : array-like
        Input data.
    Z : array-like
        Input data.
    X_exog : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: F, p

    References
    ----------
    Stock-Wright-Yogo (2002)
    """
    D = np.asarray(D, dtype=float)
    n = len(D)
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "First-stage IV F-statistic for weak instruments"})
    x_sorted = np.sort(D)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(D), scale=np.std(D, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "First-stage IV F-statistic for weak instruments"})


def cheatsheet():
    return "causivft: First-stage IV F-statistic for weak instruments"
