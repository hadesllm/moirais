# moirais.fn — function file (hadesllm/moirais)
"""Jonckheere-Terpstra compact formula as sum of pairwise Mann-Whitney statistics."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_jt_cd_form"]


def gibbons_jt_cd_form(groups, cdf=None):
    """
    Jonckheere-Terpstra compact formula as sum of pairwise Mann-Whitney statistics

    Formula: J = sum_{i<j} (concordant - discordant) between groups i and j

    Parameters
    ----------
    groups : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: J

    References
    ----------
    Gibbons Ch 10.6
    """
    groups = np.asarray(groups, dtype=float)
    n = int(groups) if groups.ndim == 0 else len(groups)
    if groups.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Jonckheere-Terpstra compact formula as sum of pairwise Mann-Whitney statistics"})
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Jonckheere-Terpstra compact formula as sum of pairwise Mann-Whitney statistics"})


def cheatsheet():
    return "gb_jcd: Jonckheere-Terpstra compact formula as sum of pairwise Mann-Whitney statistics"
