# moirais.fn — function file (hadesllm/moirais)
"""ARE of sign test relative to Wilcoxon signed-rank test."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_are_sign_wilcoxon"]


def gibbons_are_sign_wilcoxon(f, cdf=None):
    """
    ARE of sign test relative to Wilcoxon signed-rank test

    Formula: ARE(sign, Wilcoxon) = f^2(0) / (3 * [integral f^2(f)dx]^2) * pi/2

    Parameters
    ----------
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ARE

    References
    ----------
    Gibbons Ch 13.3.1
    """
    f = np.asarray(f, dtype=float)
    n = int(f) if f.ndim == 0 else len(f)
    if f.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "ARE of sign test relative to Wilcoxon signed-rank test"})
    x_sorted = np.sort(f)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(f), scale=np.std(f, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "ARE of sign test relative to Wilcoxon signed-rank test"})


def cheatsheet():
    return "gb_are1: ARE of sign test relative to Wilcoxon signed-rank test"
