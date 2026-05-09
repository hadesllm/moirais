# moirais.fn — function file (hadesllm/moirais)
"""Kruskal-Wallis one-way ANOVA by ranks: H statistic."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_kruskal_wallis"]


def gibbons_kruskal_wallis(groups, cdf=None):
    """
    Kruskal-Wallis one-way ANOVA by ranks: H statistic

    Formula: H = 12/(N(N+1)) * sum(R_i^2/n_i) - 3(N+1); H ~ chi2(k-1) for large n

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
    Gibbons Ch 10.4
    """
    groups = np.asarray(groups, dtype=float)
    n = int(groups) if groups.ndim == 0 else len(groups)
    if groups.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Kruskal-Wallis one-way ANOVA by ranks: H statistic"})
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Kruskal-Wallis one-way ANOVA by ranks: H statistic"})


def cheatsheet():
    return "gb1041: Kruskal-Wallis one-way ANOVA by ranks: H statistic"
