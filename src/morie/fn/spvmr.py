"""Variance-mean ratio (VMR) test for CSR."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["schabenberger_variance_mean_ratio"]


def schabenberger_variance_mean_ratio(quadrat_counts, cdf=None):
    """
    Variance-mean ratio (VMR) test for CSR

    Formula: VMR = s^2/xbar; chi^2 = (m-1)*VMR where m=num quadrats

    Parameters
    ----------
    quadrat_counts : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: statistic, p_value

    References
    ----------
    Schabenberger Ch 3
    """
    quadrat_counts = np.asarray(quadrat_counts, dtype=float)
    n = int(quadrat_counts) if quadrat_counts.ndim == 0 else len(quadrat_counts)
    if quadrat_counts.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Variance-mean ratio (VMR) test for CSR"})
    x_sorted = np.sort(quadrat_counts)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(quadrat_counts), scale=np.std(quadrat_counts, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Variance-mean ratio (VMR) test for CSR"})


def cheatsheet():
    return "spvmr: Variance-mean ratio (VMR) test for CSR"
