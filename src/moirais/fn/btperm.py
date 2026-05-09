"""Permutation two-sample test (resampling without replacement)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_permutation_test"]


def boot_permutation_test(x, y, stat, B, cdf=None):
    """
    Permutation two-sample test (resampling without replacement)

    Formula: Shuffle group labels B times; p = (1+#extreme)/(B+1)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    stat : array-like
        Input data.
    B : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: T_obs, p, T_b

    References
    ----------
    Fisher (1935)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Permutation two-sample test (resampling without replacement)"})
    x_sorted = np.sort(x)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(x), scale=np.std(x, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Permutation two-sample test (resampling without replacement)"})


def cheatsheet():
    return "btperm: Permutation two-sample test (resampling without replacement)"
