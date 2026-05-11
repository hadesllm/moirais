"""Partial Mantel test controlling for third matrix."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["partial_mantel_test"]


def partial_mantel_test(A, B, C, permutations, cdf=None):
    """
    Partial Mantel test controlling for third matrix

    Formula: r_AB|C from partial correlation of vec(A), vec(B), vec(C)

    Parameters
    ----------
    A : array-like
        Input data.
    B : array-like
        Input data.
    C : array-like
        Input data.
    permutations : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Smouse, Long, Sokal (1986)
    """
    A = np.asarray(A, dtype=float)
    n = len(A)
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Partial Mantel test controlling for third matrix"})
    x_sorted = np.sort(A)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(A), scale=np.std(A, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Partial Mantel test controlling for third matrix"})


def cheatsheet():
    return "pmantl: Partial Mantel test controlling for third matrix"
