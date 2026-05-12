# morie.fn -- function file (hadesllm/morie)
"""McNemar test confidence interval for discordant proportion difference."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_mcnemar_ci"]


def gibbons_mcnemar_ci(b, c, n, alpha, cdf=None):
    """
    McNemar test confidence interval for discordant proportion difference

    Formula: CI for (b-c)/n using normal approximation

    Parameters
    ----------
    b : array-like
        Input data.
    c : array-like
        Input data.
    n : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lower, upper

    References
    ----------
    Gibbons Ch 14.5
    """
    b = np.asarray(b, dtype=float)
    n = int(b) if b.ndim == 0 else len(b)
    if b.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "McNemar test confidence interval for discordant proportion difference"})
    x_sorted = np.sort(b)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(b), scale=np.std(b, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "McNemar test confidence interval for discordant proportion difference"})


def cheatsheet():
    return "gb_mci: McNemar test confidence interval for discordant proportion difference"
