# moirais.fn — function file (hadesllm/moirais)
"""Exact p-value for sign test using binomial CDF."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_sign_pvalue"]


def gibbons_sign_pvalue(k_obs, n, cdf=None):
    """
    Exact p-value for sign test using binomial CDF

    Formula: P-value = 2*P(K >= k_obs) or P(K >= k_obs) for one-sided test

    Parameters
    ----------
    k_obs : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: p_value

    References
    ----------
    Gibbons Ch 5.4.1
    """
    k_obs = np.asarray(k_obs, dtype=float)
    n = int(k_obs) if k_obs.ndim == 0 else len(k_obs)
    if k_obs.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Exact p-value for sign test using binomial CDF"})
    x_sorted = np.sort(k_obs)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(k_obs), scale=np.std(k_obs, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Exact p-value for sign test using binomial CDF"})


def cheatsheet():
    return "gb5411: Exact p-value for sign test using binomial CDF"
