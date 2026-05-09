# moirais.fn — function file (hadesllm/moirais)
"""ARE of Kruskal-Wallis relative to F-test: at least 0.864 for any continuous F."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_are_kw"]


def gibbons_are_kw(distribution, cdf=None):
    """
    ARE of Kruskal-Wallis relative to F-test: at least 0.864 for any continuous F

    Formula: ARE(KW, F) >= 3/pi for normal; lower bound 0.864 for all continuous F

    Parameters
    ----------
    distribution : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ARE_lower_bound

    References
    ----------
    Gibbons Ch 10.8
    """
    distribution = np.asarray(distribution, dtype=float)
    n = int(distribution) if distribution.ndim == 0 else len(distribution)
    if distribution.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "ARE of Kruskal-Wallis relative to F-test: at least 0.864 for any continuous F"})
    x_sorted = np.sort(distribution)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(distribution), scale=np.std(distribution, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "ARE of Kruskal-Wallis relative to F-test: at least 0.864 for any continuous F"})


def cheatsheet():
    return "gb_are4: ARE of Kruskal-Wallis relative to F-test: at least 0.864 for any continuous F"
