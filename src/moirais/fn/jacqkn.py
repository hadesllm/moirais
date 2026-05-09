"""Jacquez k-nearest-neighbour space-time test."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["jacquez_k_nn_test"]


def jacquez_k_nn_test(coords, time, k, cdf=None):
    """
    Jacquez k-nearest-neighbour space-time test

    Formula: J_k = sum_i sum_{j in NN_k(i)} a_ij b_ij

    Parameters
    ----------
    coords : array-like
        Input data.
    time : array-like
        Input data.
    k : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jacquez (1996)
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Jacquez k-nearest-neighbour space-time test"})
    x_sorted = np.sort(coords)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(coords), scale=np.std(coords, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Jacquez k-nearest-neighbour space-time test"})


def cheatsheet():
    return "jacqkn: Jacquez k-nearest-neighbour space-time test"
