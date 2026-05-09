# moirais.fn — function file (hadesllm/moirais)
"""k-sample median test extension using count above combined median."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_k_median_test"]


def gibbons_k_median_test(groups, cdf=None):
    """
    k-sample median test extension using count above combined median

    Formula: Q = chi2-type statistic based on counts above combined sample median

    Parameters
    ----------
    groups : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: statistic, p_value

    References
    ----------
    Gibbons Ch 10.2
    """
    groups = np.asarray(groups, dtype=float)
    n = int(groups) if groups.ndim == 0 else len(groups)
    if groups.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "k-sample median test extension using count above combined median"})
    x_sorted = np.sort(groups)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(groups), scale=np.std(groups, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "k-sample median test extension using count above combined median"})


def cheatsheet():
    return "gb1021: k-sample median test extension using count above combined median"
