"""Test-negative design VE."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["test_negative_design"]


def test_negative_design(test_status, vaccination_status, cdf=None):
    """
    Test-negative design VE

    Formula: OR among test-positives vs test-negatives

    Parameters
    ----------
    test_status : array-like
        Input data.
    vaccination_status : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jackson-Nelson (2013)
    """
    test_status = np.asarray(test_status, dtype=float)
    n = len(test_status)
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Test-negative design VE"})
    x_sorted = np.sort(test_status)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(test_status), scale=np.std(test_status, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Test-negative design VE"})


def cheatsheet():
    return "tnvest: Test-negative design VE"
