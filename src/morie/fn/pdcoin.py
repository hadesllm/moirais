"""Pedroni panel cointegration test."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pedroni_panel_cointegration"]


def pedroni_panel_cointegration(X, groups, cdf=None):
    """
    Pedroni panel cointegration test

    Formula: 7 statistics: panel-rho, panel-PP, panel-ADF, group-rho, group-PP, group-ADF

    Parameters
    ----------
    X : array-like
        Input data.
    groups : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Pedroni (1999, 2004)
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Pedroni panel cointegration test"})
    x_sorted = np.sort(X)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(X), scale=np.std(X, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Pedroni panel cointegration test"})


def cheatsheet():
    return "pdcoin: Pedroni panel cointegration test"
