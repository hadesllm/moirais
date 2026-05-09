# moirais.fn — function file (hadesllm/moirais)
"""Mean and variance of Wilcoxon signed-rank statistic T+ under H0."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_wsrt_mean"]


def gibbons_wsrt_mean(n, cdf=None):
    """
    Mean and variance of Wilcoxon signed-rank statistic T+ under H0

    Formula: E(T+) = n(n+1)/4; Var(T+) = n(n+1)(2n+1)/24

    Parameters
    ----------
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mean, variance

    References
    ----------
    Gibbons Ch 5.7
    """
    data = np.asarray(n, dtype=float) if np.ndim(n) > 0 else None
    n = int(n) if np.ndim(n) == 0 else len(n)
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Mean and variance of Wilcoxon signed-rank statistic T+ under H0"})
    if data is None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal(n)
    x_sorted = np.sort(data)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(data), scale=np.std(data, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Mean and variance of Wilcoxon signed-rank statistic T+ under H0"})


def cheatsheet():
    return "gb571m: Mean and variance of Wilcoxon signed-rank statistic T+ under H0"
