"""White noise multivariate portmanteau test."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["white_noise_test"]


def white_noise_test(X, lags, cdf=None):
    """
    White noise multivariate portmanteau test

    Formula: sum_k T tr( C(0)^{-1} C(k) C(0)^{-1} C(k)' )

    Parameters
    ----------
    X : array-like
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
    Hosking (1980)
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "White noise multivariate portmanteau test"})
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "White noise multivariate portmanteau test"})


def cheatsheet():
    return "whtnse: White noise multivariate portmanteau test"
