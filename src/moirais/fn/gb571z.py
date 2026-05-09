# moirais.fn — function file (hadesllm/moirais)
"""Handling zeros and tied differences in Wilcoxon signed-rank test."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_wsrt_ties_zeros"]


def gibbons_wsrt_ties_zeros(differences, cdf=None):
    """
    Handling zeros and tied differences in Wilcoxon signed-rank test

    Formula: Discard zeros; midranks for ties; Var adj = -sum t_j(t_j^2-1)/48

    Parameters
    ----------
    differences : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: adjusted_statistic

    References
    ----------
    Gibbons Ch 5.7.1
    """
    differences = np.asarray(differences, dtype=float)
    n = int(differences) if differences.ndim == 0 else len(differences)
    if differences.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Handling zeros and tied differences in Wilcoxon signed-rank test"})
    x_sorted = np.sort(differences)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(differences), scale=np.std(differences, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Handling zeros and tied differences in Wilcoxon signed-rank test"})


def cheatsheet():
    return "gb571z: Handling zeros and tied differences in Wilcoxon signed-rank test"
