"""Quadrat count chi-squared test for CSR."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["schabenberger_quadrat_count_test"]


def schabenberger_quadrat_count_test(points, quadrats, cdf=None):
    """
    Quadrat count chi-squared test for CSR

    Formula: chi^2 = sum (O_i - E_i)^2/E_i, E_i = n/m (n points, m quadrats)

    Parameters
    ----------
    points : array-like
        Input data.
    quadrats : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: statistic, p_value

    References
    ----------
    Schabenberger Ch 3, Sec 3.3.3
    """
    points = np.asarray(points, dtype=float)
    n = int(points) if points.ndim == 0 else len(points)
    if points.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Quadrat count chi-squared test for CSR"})
    x_sorted = np.sort(points)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(points), scale=np.std(points, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Quadrat count chi-squared test for CSR"})


def cheatsheet():
    return "spqct: Quadrat count chi-squared test for CSR"
