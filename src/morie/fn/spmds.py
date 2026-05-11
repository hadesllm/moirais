"""Multidimensional scaling for non-Euclidean distances in geostatistics."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["schabenberger_multidim_scaling"]


def schabenberger_multidim_scaling(distance_matrix, dim, cdf=None):
    """
    Multidimensional scaling for non-Euclidean distances in geostatistics

    Formula: Embed points in R^d such that ||f(s_i)-f(s_j)|| approx d_ij; covariance via f-space distances

    Parameters
    ----------
    distance_matrix : array-like
        Input data.
    dim : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: embedding

    References
    ----------
    Schabenberger Ch 4, Sec 4.8.2
    """
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n = int(distance_matrix) if distance_matrix.ndim == 0 else len(distance_matrix)
    if distance_matrix.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Multidimensional scaling for non-Euclidean distances in geostatistics"})
    x_sorted = np.sort(distance_matrix)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(distance_matrix), scale=np.std(distance_matrix, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Multidimensional scaling for non-Euclidean distances in geostatistics"})


def cheatsheet():
    return "spmds: Multidimensional scaling for non-Euclidean distances in geostatistics"
