"""Engle ARCH test for heteroscedasticity."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["arch_test"]


def arch_test(residuals, lags, cdf=None):
    """
    Engle ARCH test for heteroscedasticity

    Formula: LM stat from regression of e_t^2 on lags

    Parameters
    ----------
    residuals : array-like
        Input data.
    lags : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Engle (1982)
    """
    residuals = np.asarray(residuals, dtype=float)
    n = len(residuals)
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Engle ARCH test for heteroscedasticity"})
    x_sorted = np.sort(residuals)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(residuals), scale=np.std(residuals, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Engle ARCH test for heteroscedasticity"})


def cheatsheet():
    return "archtt: Engle ARCH test for heteroscedasticity"
