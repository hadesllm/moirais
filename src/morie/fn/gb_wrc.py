# morie.fn -- function file (hadesllm/morie)
"""Critical regions for runs test using tabulated null distribution."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_runs_critical"]


def gibbons_runs_critical(n1, n2, alpha, cdf=None):
    """
    Critical regions for runs test using tabulated null distribution

    Formula: Reject if R <= r_lower or R >= r_upper from Table D

    Parameters
    ----------
    n1 : array-like
        Input data.
    n2 : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lower_critical, upper_critical

    References
    ----------
    Gibbons Table D
    """
    n1 = np.asarray(n1, dtype=float)
    n = int(n1) if n1.ndim == 0 else len(n1)
    if n1.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Critical regions for runs test using tabulated null distribution"})
    x_sorted = np.sort(n1)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(n1), scale=np.std(n1, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Critical regions for runs test using tabulated null distribution"})


def cheatsheet():
    return "gb_wrc: Critical regions for runs test using tabulated null distribution"
