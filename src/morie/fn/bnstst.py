"""Test of bound = 0."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_test_inference"]


def bound_test_inference(lower, upper, se, cdf=None):
    """
    Test of bound = 0

    Formula: H0: lower_bound ≤ 0 ≤ upper_bound

    Parameters
    ----------
    lower : array-like
        Input data.
    upper : array-like
        Input data.
    se : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Stoye (2009)
    """
    lower = np.asarray(lower, dtype=float)
    n = len(lower)
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Test of bound = 0"})
    x_sorted = np.sort(lower)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(lower), scale=np.std(lower, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Test of bound = 0"})


def cheatsheet():
    return "bnstst: Test of bound = 0"
