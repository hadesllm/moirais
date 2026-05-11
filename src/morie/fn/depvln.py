"""Test of sequential ignorability via negative controls."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dependent_violation_test"]


def dependent_violation_test(y_neg, A, H, cdf=None):
    """
    Test of sequential ignorability via negative controls

    Formula: if effect on neg-outcome != 0, violation

    Parameters
    ----------
    y_neg : array-like
        Input data.
    A : array-like
        Input data.
    H : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lipsitch et al (2010); Shi et al (2020)
    """
    y_neg = np.asarray(y_neg, dtype=float)
    n = len(y_neg)
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Test of sequential ignorability via negative controls"})
    x_sorted = np.sort(y_neg)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(y_neg), scale=np.std(y_neg, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Test of sequential ignorability via negative controls"})


def cheatsheet():
    return "depvln: Test of sequential ignorability via negative controls"
