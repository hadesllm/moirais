# moirais.fn — function file (hadesllm/moirais)
"""Mean and variance of Jonckheere-Terpstra statistic under H0."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_jt_moments"]


def gibbons_jt_moments(n_i, cdf=None):
    """
    Mean and variance of Jonckheere-Terpstra statistic under H0

    Formula: E(J) = N^2 - sum n_i^2)/4; complex variance formula

    Parameters
    ----------
    n_i : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mean, variance

    References
    ----------
    Gibbons Ch 10.6
    """
    n_i = np.asarray(n_i, dtype=float)
    n = int(n_i) if n_i.ndim == 0 else len(n_i)
    if n_i.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Mean and variance of Jonckheere-Terpstra statistic under H0"})
    x_sorted = np.sort(n_i)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(n_i), scale=np.std(n_i, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Mean and variance of Jonckheere-Terpstra statistic under H0"})


def cheatsheet():
    return "gb1061m: Mean and variance of Jonckheere-Terpstra statistic under H0"
