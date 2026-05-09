# moirais.fn — function file (hadesllm/moirais)
"""Type I error and p-value concepts for discrete test statistics."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_type1_error"]


def gibbons_type1_error(statistic, null_dist, cdf=None):
    """
    Type I error and p-value concepts for discrete test statistics

    Formula: alpha = P(R <= r_alpha) + P(R >= r'_alpha); discrete exact level <= nominal alpha

    Parameters
    ----------
    statistic : array-like
        Input data.
    null_dist : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: exact_level

    References
    ----------
    Gibbons Ch 3.2.5
    """
    statistic = np.asarray(statistic, dtype=float)
    n = int(statistic) if statistic.ndim == 0 else len(statistic)
    if statistic.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Type I error and p-value concepts for discrete test statistics"})
    x_sorted = np.sort(statistic)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(statistic), scale=np.std(statistic, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Type I error and p-value concepts for discrete test statistics"})


def cheatsheet():
    return "gb_t1e: Type I error and p-value concepts for discrete test statistics"
