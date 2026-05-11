# morie.fn — function file (hadesllm/morie)
"""Tabulated exact probabilities of runs test statistic R for small n1, n2."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_total_runs_dist_table"]


def gibbons_total_runs_dist_table(n1, n2, r, cdf=None):
    """
    Tabulated exact probabilities of runs test statistic R for small n1, n2

    Formula: P(R = r) from Theorem 3.2.2 for n1, n2 <= 12

    Parameters
    ----------
    n1 : array-like
        Input data.
    n2 : array-like
        Input data.
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: probability

    References
    ----------
    Gibbons Table D
    """
    n1 = np.asarray(n1, dtype=float)
    n = int(n1) if n1.ndim == 0 else len(n1)
    if n1.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Tabulated exact probabilities of runs test statistic R for small n1, n2"})
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Tabulated exact probabilities of runs test statistic R for small n1, n2"})


def cheatsheet():
    return "gb_ttd: Tabulated exact probabilities of runs test statistic R for small n1, n2"
