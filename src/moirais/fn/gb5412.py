# moirais.fn — function file (hadesllm/moirais)
"""Normal approximation to sign test with continuity correction."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_sign_normal_approx"]


def gibbons_sign_normal_approx(k, n, cdf=None):
    """
    Normal approximation to sign test with continuity correction

    Formula: Z = (K - n/2) / (sqrt(n)/2); continuity correction: (K - 0.5 - n/2)/(sqrt(n)/2)

    Parameters
    ----------
    k : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: z_statistic

    References
    ----------
    Gibbons Ch 5.4.2
    """
    k = np.asarray(k, dtype=float)
    n = int(k) if k.ndim == 0 else len(k)
    if k.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Normal approximation to sign test with continuity correction"})
    x_sorted = np.sort(k)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(k), scale=np.std(k, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Normal approximation to sign test with continuity correction"})


def cheatsheet():
    return "gb5412: Normal approximation to sign test with continuity correction"
