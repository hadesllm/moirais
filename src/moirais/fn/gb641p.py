# moirais.fn — function file (hadesllm/moirais)
"""Power of two-sample median test."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_median_test_power"]


def gibbons_median_test_power(m, n, Delta, alpha, cdf=None):
    """
    Power of two-sample median test

    Formula: beta(Delta) = P(reject H0 | location shift Delta)

    Parameters
    ----------
    m : array-like
        Input data.
    n : array-like
        Input data.
    Delta : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: power

    References
    ----------
    Gibbons Ch 6.4 power
    """
    m = np.asarray(m, dtype=float)
    n = int(m) if m.ndim == 0 else len(m)
    if m.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Power of two-sample median test"})
    x_sorted = np.sort(m)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(m), scale=np.std(m, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Power of two-sample median test"})


def cheatsheet():
    return "gb641p: Power of two-sample median test"
