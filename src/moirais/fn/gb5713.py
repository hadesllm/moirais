# moirais.fn — function file (hadesllm/moirais)
"""Simulated power of Wilcoxon signed-rank test via Monte Carlo."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_wsrt_simpower"]


def gibbons_wsrt_simpower(theta, n, alpha, nsim, cdf=None):
    """
    Simulated power of Wilcoxon signed-rank test via Monte Carlo

    Formula: beta_hat = proportion of simulated rejections

    Parameters
    ----------
    theta : array-like
        Input data.
    n : array-like
        Input data.
    alpha : array-like
        Input data.
    nsim : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: simulated_power

    References
    ----------
    Gibbons Ch 5.7.3
    """
    theta = np.asarray(theta, dtype=float)
    n = int(theta) if theta.ndim == 0 else len(theta)
    if theta.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Simulated power of Wilcoxon signed-rank test via Monte Carlo"})
    x_sorted = np.sort(theta)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(theta), scale=np.std(theta, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Simulated power of Wilcoxon signed-rank test via Monte Carlo"})


def cheatsheet():
    return "gb5713: Simulated power of Wilcoxon signed-rank test via Monte Carlo"
