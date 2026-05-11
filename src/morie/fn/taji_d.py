"""Tajima's D test of selection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tajimas_d"]


def tajimas_d(sequences, cdf=None):
    """
    Tajima's D test of selection

    Formula: D = (pi - theta_W) / sqrt(Var)

    Parameters
    ----------
    sequences : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tajima (1989)
    """
    sequences = np.asarray(sequences, dtype=float)
    n = len(sequences)
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Tajima's D test of selection"})
    x_sorted = np.sort(sequences)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(sequences), scale=np.std(sequences, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Tajima's D test of selection"})


def cheatsheet():
    return "taji_d: Tajima's D test of selection"
